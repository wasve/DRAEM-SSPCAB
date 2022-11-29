# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import mindspore.nn as nn
import mindspore.ops as ops
import mindspore as ms
from math import exp


def gaussian():
    gauss = ms.Tensor([exp(-(x - 11 // 2) ** 2 / float(2 * 1.5 ** 2)) for x in range(11)],
                      ms.float32)
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    matmul = ops.MatMul()
    _1D_window = ops.expand_dims(gaussian(), 1)
    _2D_window = ops.expand_dims(ops.expand_dims(matmul(_1D_window, _1D_window.T), 0), 0)
    window = ops.broadcast_to(_2D_window, (channel, 1, window_size, window_size))
    return window


class SSIMThen(nn.Cell):
    def __init__(self):
        super(SSIMThen, self).__init__()
        self._001 = ms.Tensor(0.01, ms.float32)
        self._003 = ms.Tensor(0.03, ms.float32)
        self._255 = ms.Tensor(255, ms.int32)
        self._1 = ms.Tensor(1, ms.int32)
        self._neg_1 = ms.Tensor(-1, ms.int32)
        self._zero = ms.Tensor(0, ms.int32)

    def construct(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        if img1.max() > 128:
            max_val = self._255
        else:
            max_val = self._1
        if img1.min() < -0.5:
            min_val = self._neg_1
        else:
            min_val = self._zero
        l = max_val - min_val
        padd = window_size // 2
        (_, channel, height, width) = img1.shape
        if window is None:
            real_size = min(window_size, height, width)
            window = create_window(real_size, channel=channel)
        cov2d = ops.Conv2D(out_channel=window.shape[0],
                           kernel_size=(window.shape[2], window.shape[3]),
                           pad_mode='pad',
                           pad=(padd, padd, padd, padd),
                           group=channel)
        mu1 = cov2d(img1, window)
        mu2 = cov2d(img2, window)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cov2d(img1 * img1, window) - mu1_sq
        sigma2_sq = cov2d(img2 * img2, window) - mu2_sq
        sigma12 = cov2d(img1 * img2, window) - mu1_mu2
        c1 = (self._001 * l) ** 2
        c2 = (self._003 * l) ** 2

        v1 = 2.0 * sigma12 + c2
        v2 = sigma1_sq + sigma2_sq + c2
        cs = ops.mean(v1 / v2)
        ssim_map = ((2 * mu1_mu2 + c1) * v1) / ((mu1_sq + mu2_sq + c1) * v2)
        ret = ssim_map.mean()
        return ret, ssim_map


class MindSSIM(nn.Cell):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(MindSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.ssim = SSIMThen()

    def construct(self, img1, img2):
        (_, channel, _, _) = img1.shape
        window = create_window(self.window_size, channel)
        s_score, ssim_map = self.ssim(img1, img2, window=window, window_size=self.window_size,
                                      size_average=self.size_average)
        return 1.0 - s_score


class MindFocalLoss(nn.Cell):
    def __init__(self, apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True):
        super(MindFocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.transpose = ops.Transpose()
        self.squeeze = ops.Squeeze(1)
        self.ones = ops.Ones()
        self.zeros = ops.Zeros()
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def construct(self, logit: ms.Tensor, target: ms.Tensor):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]
        if len(logit.shape) > 2:
            logit = logit.view(logit.shape[0], logit.shape[1], -1)
            logit = self.transpose(logit, (0, 2, 1))
            logit = logit.view(-1, logit.shape[-1])
        target = self.squeeze(target)
        target = target.view(-1, 1)
        alpha = self.alpha
        alpha = self.ones((num_class, 1), ms.float32)
        idx = target
        idx = idx.astype(ms.int64)
        onehot = ops.OneHot()
        one_hot_key = onehot(idx, 2, ms.Tensor(1, ms.int32), ms.Tensor(0.0, ms.int32))
        one_hot_key = self.squeeze(one_hot_key)
        if self.smooth:
            one_hot_key = ops.clip_by_value(
                one_hot_key, ms.Tensor(self.smooth / (num_class - 1)), ms.Tensor(1.0 - self.smooth))
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = ops.log(pt)
        gamma = self.gamma
        alpha = alpha[idx]
        alpha = ops.squeeze(alpha)
        loss = -1 * alpha * ops.pow((1 - pt), gamma) * logpt
        if self.size_average:
            loss = loss.mean()
        return loss
