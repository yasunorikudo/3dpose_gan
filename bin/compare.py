import argparse
import chainer
import cv2
import json
import numpy as np
import os
from tqdm import tqdm

from bin.evaluation_util import create_img
from bin.evaluation_util import create_projection_img
from projection_gan.pose.posenet import MLP
from projection_gan.pose.dataset.pose_dataset import MPII

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_sup', type=str)
    parser.add_argument('model_unsup', type=str)
    parser.add_argument('--mpii_images', type=str, default='data/mpii_image_name.txt')
    parser.add_argument('--mpii_detections', type=str, default='data/mpii_poses.npy')
    parser.add_argument('--mpii_root', type=str, required=True)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--small_dataset', type=int, default=-1)
    args = parser.parse_args()

    with open(args.mpii_images) as f:
        mpii_images = f.read().strip().split('\n')
    mpii_detections = np.load(args.mpii_detections)

    with open(os.path.join(os.path.dirname(args.model_sup), 'options.json')) as f:
        config_sup = json.load(f)
    with open(os.path.join(os.path.dirname(args.model_unsup), 'options.json')) as f:
        config_unsup = json.load(f)

    nn_sup = MLP(
        mode='generator',
        use_bn=config_sup['use_bn'],
        activate_func=getattr(chainer.functions, config_sup['activate_func']))
    nn_unsup = MLP(
        mode='generator',
        use_bn=config_unsup['use_bn'],
        activate_func=getattr(chainer.functions, config_unsup['activate_func']))

    chainer.serializers.load_npz(args.model_sup, nn_sup)
    chainer.serializers.load_npz(args.model_unsup, nn_unsup)

    test = MPII(train=False, use_sh_detection=False)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)
    perm = np.random.RandomState(seed=100).permutation(len(mpii_images))
    test_idx = perm[int(len(mpii_images)*0.9):]

    xy = np.empty((0, 34), 'f')
    z_sup = np.empty((0, 17), 'f')
    z_unsup = np.empty((0, 17), 'f')
    xp = np if args.gpu < 0 else chainer.cuda.cupy
    for batch in test_iter:
        xy_proj, _, _ = chainer.dataset.concat_examples(batch, device=args.gpu)
        xy = xp.concatenate((xy, xy_proj[:, 0, 0]), axis=0)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            z_sup = xp.concatenate((z_sup, nn_sup(xy_proj[:, 0]).data), axis=0)
            z_unsup = xp.concatenate((z_unsup, nn_unsup(xy_proj[:, 0]).data), axis=0)

    pose3d_sup = xp.reshape(xp.stack((xy[:, 0::2], xy[:, 1::2], z_sup), axis=-1), (len(xy), -1))
    pose3d_unsup = xp.reshape(xp.stack((xy[:, 0::2], xy[:, 1::2], z_unsup), axis=-1), (len(xy), -1))

    total = len(test_idx) if args.small_dataset <= 0 else args.small_dataset
    pbar = tqdm(total=total)
    for j, i in enumerate(test_idx):
        full_path = os.path.join(args.mpii_root, mpii_images[i])
        img = cv2.imread(full_path)

        deg = 15
        os.makedirs(os.path.join("data", "images", str(j)), exist_ok=True)
        for d in range(0, 360 + deg, deg):
            img_sup = create_projection_img(pose3d_sup[j:j+1], np.pi * d / 180.)
            img_sup = cv2.putText(img_sup, "supervised", (10, img_sup.shape[0]-10), 0, 0.8, (256, 0, 0), 1)
            img_unsup = create_projection_img(pose3d_unsup[j:j+1], np.pi * d / 180.)
            img_unsup = cv2.putText(img_unsup, "unsupervised", (10, img_unsup.shape[0]-10), 0, 0.8, (0, 0, 256), 1)
            concat_img = xp.concatenate((img_sup, img_unsup), axis=1)
            cv2.imwrite(os.path.join("data", "images", str(j), "rot_{:03d}_degree.png".format(d)), concat_img)

        img = create_img(mpii_detections[i], img)
        cv2.imwrite(os.path.join("data", "images", str(j), "input.jpg"), img)
        pbar.update(1)
        if args.small_dataset > 0 and j + 1 == args.small_dataset:
            break
    pbar.close()
