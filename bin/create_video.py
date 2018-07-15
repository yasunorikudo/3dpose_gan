#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 Yasunori Kudo

import argparse
import cv2
import json
import numpy as np
import os
import pickle
from progressbar import ProgressBar
import subprocess
import sys
sys.path.append(os.getcwd())

import chainer
import chainer.functions as F
from chainer import serializers
from chainer import Variable

from projection_gan.pose.posenet import MLP
from projection_gan.pose.updater import H36M_Updater
from projection_gan.pose.dataset.pose_dataset import H36M

from evaluation_util import color_jet, create_img

if __name__ == '__main__':
    print(subprocess.check_output(['pyenv', 'version']).decode('utf-8').strip())

    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--row', type=int, default=6)
    parser.add_argument('--col', type=int, default=6)
    parser.add_argument('--action', '-a', type=str, default='')
    parser.add_argument('--image', action='store_true')
    args = parser.parse_args()

    # Load parser options.
    with open(os.path.join(os.path.dirname(args.model_path), 'options.json')) as f:
        j = json.load(f)
    class Empty: pass
    opts = Empty()
    for k, v in j.items():
        setattr(opts, k, v)

    action = args.action if args.action else opts.action

    spr = 3  # second per rotatation
    fps = 20  # frame per second
    n_rot = 3  # number of rotation
    l_seq = spr * fps * n_rot
    imgs = np.zeros((l_seq, 350 * args.col, 600 * args.row, 3), dtype=np.uint8)

    gen = MLP(mode='generator', use_bn=opts.use_bn,
              activate_func=getattr(F, opts.activate_func))
    serializers.load_npz(args.model_path, gen)

    val = H36M(action=action, train=False,
               use_sh_detection=True)
    val_iter = chainer.iterators.SerialIterator(
        val, batch_size=args.row, shuffle=True, repeat=False)

    chainer.config.train = False
    chainer.config.enable_backprop = False
    chainer.config.show()

    pbar = ProgressBar(args.col)
    for k in range(args.col):
        batch = val_iter.next()
        batch = chainer.dataset.concat_examples(batch)
        xy_proj, xyz, scale = batch

        xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]

        xy_real = Variable(xy_proj)
        z_pred = gen(xy_real)

        for i, theta in enumerate(gen.xp.arange(0, 2 * np.pi, 2 * np.pi / fps / spr)):
            cos_theta = gen.xp.cos(gen.xp.array([theta] * args.row, 'f'))[:, None]
            sin_theta = gen.xp.sin(gen.xp.array([theta] * args.row, 'f'))[:, None]

            # 2D Projection.
            x = xy_real[:, 0::2]
            y = xy_real[:, 1::2]
            new_x = x * cos_theta + z_pred * sin_theta
            xy_fake = F.concat((new_x[:, :, None], y[:, :, None]), axis=2)
            xy_fake = F.reshape(xy_fake, (args.row, -1))

            xx = xyz[:, 0::3]
            yy = xyz[:, 1::3]
            zz = xyz[:, 2::3]
            new_xx = xx * cos_theta + zz * sin_theta
            xy_real2 = F.concat((new_xx[:, :, None], yy[:, :, None]), axis=2)
            xy_real2 = F.reshape(xy_real2, (args.row, -1))

            for j in range(args.row):
                im0 = create_img(xy_real.data[j])
                im1 = create_img(xy_real2.data[j])
                im2 = create_img(xy_fake.data[j])

                if j == 0 and k == 0:
                    im0 = cv2.putText(im0, 'Input 2D pose', (8, im0.shape[0]-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
                    im1 = cv2.putText(im1, 'GT 3D pose', (8, im0.shape[0]-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)
                    im2 = cv2.putText(im2, 'Pred 3D pose', (8, im0.shape[0]-10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 2)

                img = np.concatenate((im0, im1, im2), axis=1)
                frame_indices = [i + r * fps * spr for r in range(n_rot)]
                imgs[frame_indices, k*350:(k+1)*350, j*600:(j+1)*600] = img
            pbar.update(k + 1)


    os.makedirs(os.path.join(os.path.dirname(args.model_path), 'videos'), exist_ok=True)
    video_path = os.path.join(os.path.dirname(args.model_path), 'videos',
                              os.path.basename(args.model_path).replace('.npz', '_action_{}.mp4'.format(action)))
    for img in imgs:
        for k in range(args.col + 1):
            img = cv2.line(img, (0, k * 350), (args.row * 600, k * 350), (0, 0, 255), 4)
        for j in range(args.row + 1):
            img = cv2.line(img, (j * 600, 0), (j * 600, args.col * 350), (0, 0, 255), 4)
    if not args.image:
        print('Saving video ...')
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (imgs.shape[2], imgs.shape[1]))
        for img in imgs:
            out.write(img)
        out.release()
        print('Saved video as \'{}\'.'.format(video_path))
    else:
        image_path = video_path.replace('.mp4', '.png')
        cv2.imwrite(image_path, imgs[0])
        print('Saved image as \'{}\'.'.format(image_path))
