#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2018 Yasunori Kudo

from __future__ import print_function

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np


class H36M_Updater(chainer.training.StandardUpdater):

    def __init__(self, gan_accuracy_cap, use_heuristic_loss,
                 heuristic_loss_weight, mode, use_camera_rotation_parameter,
                 *args, **kwargs):
        if not mode in ['supervised', 'unsupervised']:
            raise ValueError("only 'supervised' and 'unsupervised' are valid "
                             "for 'mode', but '{}' is given.".format(mode))
        self.gan_accuracy_cap = gan_accuracy_cap
        self.use_heuristic_loss = use_heuristic_loss
        self.heuristic_loss_weight = heuristic_loss_weight
        self.mode = mode
        self.use_camera_rotation_parameter = use_camera_rotation_parameter
        super(H36M_Updater, self).__init__(*args, **kwargs)

    @staticmethod
    def calculate_rotation(xy_real, z_pred):
        xy_split = F.split_axis(xy_real, xy_real.data.shape[1], axis=1)
        z_split = F.split_axis(z_pred, z_pred.data.shape[1], axis=1)
        # Vector v0 (neck -> nose) on zx-plain. v0=(a0, b0).
        a0 = z_split[9] - z_split[8]
        b0 = xy_split[9 * 2] - xy_split[8 * 2]
        n0 = F.sqrt(a0 * a0 + b0 * b0)
        # Vector v1 (right shoulder -> left shoulder) on zx-plain. v1=(a1, b1).
        a1 = z_split[14] - z_split[11]
        b1 = xy_split[14 * 2] - xy_split[11 * 2]
        n1 = F.sqrt(a1 * a1 + b1 * b1)
        # Return sine value of the angle between v0 and v1.
        return (a0 * b1 - a1 * b0) / (n0 * n1)

    @staticmethod
    def calculate_heuristic_loss(xy_real, z_pred):
        return F.average(F.relu(
            -H36M_Updater.calculate_rotation(xy_real, z_pred)))

    @staticmethod
    def create_rotation_matrix(vec, theta):
        """
        Create rotation matrix
        Args
        vec(ndarray): Batchsize x 3, Unit vector of axis
        theta(ndarray): Batchsize x 1, Radial rotation value
        Returns
        R(ndarray): Batchsize x 3 x 3, rotation matrix
        """
        batchsize = len(vec)
        xp = chainer.cuda.get_array_module(vec)
        R = xp.empty((batchsize, 3, 3), dtype='f')
        vx, vy, vz = vec.T
        cos_t = xp.cos(theta[:, 0])
        sin_t = xp.sin(theta[:, 0])
        R[:, 0, 0] = vx * vx * (1 - cos_t) + cos_t
        R[:, 0, 1] = vx * vy * (1 - cos_t) - vz * sin_t
        R[:, 0, 2] = vz * vx * (1 - cos_t) + vy * sin_t

        R[:, 1, 0] = vx * vy * (1 - cos_t) + vz * sin_t
        R[:, 1, 1] = vy * vy * (1 - cos_t) + cos_t
        R[:, 1, 2] = vy * vz * (1 - cos_t) - vx * sin_t

        R[:, 2, 0] = vz * vx * (1 - cos_t) - vy * sin_t
        R[:, 2, 1] = vy * vz * (1 - cos_t) + vx * sin_t
        R[:, 2, 2] = vz * vz * (1 - cos_t) + cos_t

        return R

    @staticmethod
    def differentiable_rotation(X, Y, Z, vec, theta):
        """
        Differentiable rotation around given axis
        Args
        X(chainer.Variable): Batchsize x N, X coordinates
        Y(chainer.Variable): Batchsize x N, Y coordinates
        Z(chainer.Variable): Batchsize x N, Z coordinates
        vec(ndarray): Batchsize x 3, Unit vector of axis
        theta(ndarray): Batchsize x 1, Radial rotation value
        Returns
        X2(chainer.Variable): Batchsize x N, X coordinates after rotation
        Y2(chainer.Variable): Batchsize x N, Y coordinates after rotation
        Z2(chainer.Variable): Batchsize x N, Z coordinates after rotation
        """
        R = H36M_Updater.create_rotation_matrix(vec, theta)
        p3d = chainer.functions.concat(
            (X[:, None], Y[:, None], Z[:, None]), axis=1)
        X2 = chainer.functions.sum(p3d * R[:, 0, :, None], axis=1)
        Y2 = chainer.functions.sum(p3d * R[:, 1, :, None], axis=1)
        Z2 = chainer.functions.sum(p3d * R[:, 2, :, None], axis=1)

        return X2, Y2, X2

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen, dis = gen_optimizer.target, dis_optimizer.target

        batch = self.get_iterator('main').next()
        batchsize = len(batch)
        xy_proj, xyz, scale = self.converter(batch, self.device)
        xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]

        xy_real = Variable(xy_proj)
        z_pred = gen(xy_real)
        z_mse = F.mean_squared_error(z_pred, xyz[:, 2::3])

        if self.mode == 'supervised':
            gen.cleargrads()
            z_mse.backward()
            gen_optimizer.update()
            chainer.report({'z_mse': z_mse}, gen)

        elif self.mode == 'unsupervised':
            # Random rotation.
            theta = gen.xp.random.uniform(0, 2 * np.pi, batchsize)
            theta = theta[:, None].astype('f')

            # 2D Projection.
            x = xy_real[:, 0::2]
            y = xy_real[:, 1::2]
            vec = gen.xp.zeros((batchsize, 3), dtype='f')
            if self.use_camera_rotation_parameter:
                vec[:, 1] = gen.xp.cos(np.pi / 2 - 1.35)  # y-axis
                vec[:, 2] = gen.xp.sin(np.pi / 2 - 1.35)  # z-axis
            else:
                vec[:, 1] = 1  # rotation around y-axis
            xf, yf, zf = self.differentiable_rotation(x, y, z_pred, vec, theta)
            xy_fake = F.concat((xf[:, :, None], yf[:, :, None]), axis=2)
            xy_fake = F.reshape(xy_fake, (batchsize, -1))

            y_real = dis(xy_real)
            y_fake = dis(xy_fake)

            acc_dis_fake = F.binary_accuracy(
                y_fake, dis.xp.zeros(y_fake.data.shape, dtype=int))
            acc_dis_real = F.binary_accuracy(
                y_real, dis.xp.ones(y_real.data.shape, dtype=int))
            acc_dis = (acc_dis_fake + acc_dis_real) / 2

            loss_gen = F.sum(F.softplus(-y_fake)) / batchsize
            if self.use_heuristic_loss:
                loss_heuristic = self.calculate_heuristic_loss(
                    xy_real=xy_real, z_pred=z_pred)
                loss_gen += loss_heuristic * self.heuristic_loss_weight
                chainer.report({'loss_heuristic': loss_heuristic}, gen)
            gen.cleargrads()
            if acc_dis.data >= (1 - self.gan_accuracy_cap):
                loss_gen.backward()
                gen_optimizer.update()
            xy_fake.unchain_backward()

            loss_dis = F.sum(F.softplus(-y_real)) / batchsize
            loss_dis += F.sum(F.softplus(y_fake)) / batchsize
            dis.cleargrads()
            if acc_dis.data <= self.gan_accuracy_cap:
                loss_dis.backward()
                dis_optimizer.update()

            chainer.report({'loss': loss_gen, 'z_mse': z_mse}, gen)
            chainer.report({
                'loss': loss_dis, 'acc': acc_dis, 'acc/fake': acc_dis_fake,
                'acc/real': acc_dis_real}, dis)
