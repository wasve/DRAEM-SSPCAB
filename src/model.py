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


from loss import *
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.initializer import initializer, Normal
from sspcab import SSPCAB


def _make_encode_dcbl_block(in_channel: int, out_channel: int) -> nn.SequentialCell:
    return nn.SequentialCell(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=(3, 3),
                  stride=(1, 1),
                  pad_mode='pad',
                  padding=(1, 1, 1, 1),
                  dilation=(1, 1),
                  weight_init=initializer(Normal(sigma=0.02, mean=0.0), shape=[out_channel, in_channel, 3, 3],
                                          dtype=ms.float32),
                  group=1,
                  has_bias=True),
        nn.BatchNorm2d(out_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[out_channel],
                                              dtype=ms.float32)),
        nn.ReLU(),
        nn.Conv2d(out_channel,
                  out_channel,
                  kernel_size=(3, 3),
                  stride=(1, 1),
                  pad_mode='pad',
                  padding=(1, 1, 1, 1),
                  dilation=(1, 1),
                  weight_init=initializer(Normal(sigma=0.02, mean=0.0), shape=[out_channel, out_channel, 3, 3],
                                          dtype=ms.float32),
                  group=1,
                  has_bias=True),
        nn.BatchNorm2d(out_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[out_channel],
                                              dtype=ms.float32)),
        nn.ReLU())


def _make_re_decode_dcbl_block(in_channel: int, out_channel: int) -> nn.SequentialCell:
    return nn.SequentialCell(
        nn.Conv2d(in_channel,
                  in_channel,
                  kernel_size=(3, 3),
                  stride=(1, 1),
                  pad_mode='pad',
                  padding=(1, 1, 1, 1),
                  dilation=(1, 1),
                  weight_init=initializer(Normal(sigma=0.02, mean=0.0), shape=[in_channel, in_channel, 3, 3],
                                          dtype=ms.float32),
                  group=1,
                  has_bias=True),
        nn.BatchNorm2d(in_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[in_channel],
                                              dtype=ms.float32)),
        nn.ReLU(),
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=(3, 3),
                  stride=(1, 1),
                  pad_mode='pad',
                  padding=(1, 1, 1, 1),
                  dilation=(1, 1),
                  weight_init=initializer(Normal(sigma=0.02, mean=0.0), shape=[out_channel, in_channel, 3, 3],
                                          dtype=ms.float32),
                  group=1,
                  has_bias=True),
        nn.BatchNorm2d(out_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[out_channel],
                                              dtype=ms.float32)),
        nn.ReLU())


def _make_dis_decode_dcbl_block(in_channel: int, out_channel: int) -> nn.SequentialCell:
    return nn.SequentialCell(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=(3, 3),
                  stride=(1, 1),
                  pad_mode='pad',
                  padding=(1, 1, 1, 1),
                  dilation=(1, 1),
                  weight_init=initializer(Normal(sigma=0.02, mean=0.0), shape=[out_channel, in_channel, 3, 3],
                                          dtype=ms.float32),
                  group=1,
                  has_bias=True),
        nn.BatchNorm2d(out_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[out_channel],
                                              dtype=ms.float32)),
        nn.ReLU(),
        nn.Conv2d(out_channel,
                  out_channel,
                  kernel_size=(3, 3),
                  stride=(1, 1),
                  pad_mode='pad',
                  padding=(1, 1, 1, 1),
                  dilation=(1, 1),
                  weight_init=initializer(Normal(sigma=0.02, mean=0.0), shape=[out_channel, out_channel, 3, 3],
                                          dtype=ms.float32),
                  group=1,
                  has_bias=True),
        nn.BatchNorm2d(out_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[out_channel],
                                              dtype=ms.float32)),
        nn.ReLU())


def _make_sspcab_dis_decode_dcbl_block(in_channel: int, out_channel: int) -> nn.SequentialCell:
    return nn.SequentialCell(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=(3, 3),
                  stride=(1, 1),
                  pad_mode='pad',
                  padding=(1, 1, 1, 1),
                  dilation=(1, 1),
                  weight_init=initializer(Normal(sigma=0.02, mean=0.0), shape=[out_channel, in_channel, 3, 3],
                                          dtype=ms.float32),
                  group=1,
                  has_bias=True),
        nn.BatchNorm2d(out_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[out_channel],
                                              dtype=ms.float32)),
        nn.ReLU(),
        SSPCAB(out_channel),
        nn.BatchNorm2d(out_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[out_channel],
                                              dtype=ms.float32)),
        nn.ReLU())


def _make_scbl_block(in_channel: int, out_channel: int) -> nn.SequentialCell:
    return nn.SequentialCell(
        nn.Conv2d(in_channel,
                  out_channel,
                  kernel_size=(3, 3),
                  stride=(1, 1),
                  pad_mode='pad',
                  padding=(1, 1, 1, 1),
                  dilation=(1, 1),
                  weight_init=initializer(Normal(sigma=0.02, mean=0.0), shape=[out_channel, in_channel, 3, 3],
                                          dtype=ms.float32),
                  group=1,
                  has_bias=True),
        nn.BatchNorm2d(out_channel,
                       gamma_init=initializer(Normal(sigma=0.02, mean=1.0), shape=[out_channel],
                                              dtype=ms.float32)),
        nn.ReLU())


class EncoderDiscriminative(nn.Cell):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = _make_encode_dcbl_block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = _make_encode_dcbl_block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = _make_encode_dcbl_block(base_width * 2, base_width * 4)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = _make_encode_dcbl_block(base_width * 4, base_width * 8)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = _make_encode_dcbl_block(base_width * 8, base_width * 8)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block6 = _make_encode_dcbl_block(base_width * 8, base_width * 8)

    def construct(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp4(b5)
        b6 = self.block6(mp5)
        return b1, b2, b3, b4, b5, b6


class DecoderDiscriminative(nn.Cell):
    def __init__(self, base_width, out_channel=3):
        super(DecoderDiscriminative, self).__init__()
        img_size = 8
        self.upb = _make_scbl_block(base_width * 8, base_width * 8)
        self.dbb = _make_dis_decode_dcbl_block(base_width * (8 + 8), base_width * 8)
        self.up1 = _make_scbl_block(base_width * 8, base_width * 4)
        self.db1 = _make_dis_decode_dcbl_block(base_width * (4 + 8), base_width * 4)
        self.up2 = _make_scbl_block(base_width * 4, base_width * 2)
        self.db2 = _make_dis_decode_dcbl_block(base_width * (2 + 4), base_width * 2)
        self.up3 = _make_scbl_block(base_width * 2, base_width * 1)
        self.db3 = _make_dis_decode_dcbl_block(base_width * (2 + 1), base_width * 1)
        self.up4 = _make_scbl_block(base_width, base_width)
        self.db4 = _make_sspcab_dis_decode_dcbl_block(base_width * (1 + 1), base_width * 1)
        self.fin_out = nn.Conv2d(base_width, out_channel, kernel_size=3, has_bias=True)
        self.ub = ops.ResizeBilinear((img_size * 2, img_size * 2), align_corners=True)
        self.u1 = ops.ResizeBilinear((img_size * 4, img_size * 4), align_corners=True)
        self.u2 = ops.ResizeBilinear((img_size * 8, img_size * 8), align_corners=True)
        self.u3 = ops.ResizeBilinear((img_size * 16, img_size * 16), align_corners=True)
        self.u4 = ops.ResizeBilinear((img_size * 32, img_size * 32), align_corners=True)
        self.concat = ops.Concat(axis=1)

    def construct(self, b1, b2, b3, b4, b5, b6):
        upb = self.ub(b6)
        upb = self.upb(upb)
        cat_b = self.concat((upb, b5))
        db_b = self.dbb(cat_b)

        up1 = self.u1(db_b)
        up1 = self.up1(up1)
        cat1 = self.concat((up1, b4))
        db1 = self.db1(cat1)

        up2 = self.u2(db1)
        up2 = self.up2(up2)
        cat2 = self.concat((up2, b3))
        db2 = self.db2(cat2)

        up3 = self.u3(db2)
        up3 = self.up3(up3)
        cat3 = self.concat((up3, b2))
        db3 = self.db3(cat3)

        up4 = self.u4(db3)
        up4 = self.up4(up4)
        cat4 = self.concat((up4, b1))
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out


class EncoderReconstructive(nn.Cell):
    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = _make_encode_dcbl_block(in_channels, base_width)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block2 = _make_encode_dcbl_block(base_width, base_width * 2)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block3 = _make_encode_dcbl_block(base_width * 2, base_width * 4)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block4 = _make_encode_dcbl_block(base_width * 4, base_width * 8)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block5 = _make_encode_dcbl_block(base_width * 8, base_width * 8)

    def construct(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        return b5


class DecoderReconstructive(nn.Cell):
    def __init__(self, base_width, out_channel=1):
        super(DecoderReconstructive, self).__init__()
        img_size = 16
        self.up1 = _make_scbl_block(base_width * 8, base_width * 8)
        self.db1 = _make_re_decode_dcbl_block(base_width * 8, base_width * 4)
        self.up2 = _make_scbl_block(base_width * 4, base_width * 4)
        self.db2 = _make_re_decode_dcbl_block(base_width * 4, base_width * 2)
        self.up3 = _make_scbl_block(base_width * 2, base_width * 2)
        self.db3 = _make_re_decode_dcbl_block(base_width * 2, base_width * 1)
        self.up4 = _make_scbl_block(base_width, base_width)
        self.db4 = _make_re_decode_dcbl_block(base_width * 1, base_width * 1)
        self.fin_out = nn.Conv2d(base_width, out_channel, kernel_size=3, has_bias=True)
        self.u1 = ops.ResizeBilinear((img_size * 2, img_size * 2), align_corners=True)
        self.u2 = ops.ResizeBilinear((img_size * 4, img_size * 4), align_corners=True)
        self.u3 = ops.ResizeBilinear((img_size * 8, img_size * 8), align_corners=True)
        self.u4 = ops.ResizeBilinear((img_size * 16, img_size * 16), align_corners=True)

    def construct(self, x):
        x = self.u1(x)
        up1 = self.up1(x)
        db1 = self.db1(up1)
        db1 = self.u2(db1)
        up2 = self.up2(db1)
        db2 = self.db2(up2)
        db2 = self.u3(db2)
        up3 = self.up3(db2)
        db3 = self.db3(up3)
        db3 = self.u4(db3)
        up4 = self.up4(db3)
        db4 = self.db4(up4)
        out = self.fin_out(db4)
        return out


class DiscriminativeSubNetwork(nn.Cell):
    def __init__(self, in_channel=6, out_channel=2, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channel, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channel=out_channel)
        self.out_features = out_features

    def construct(self, x):
        b1, b2, b3, b4, b5, b6 = self.encoder_segment(x)

        output_segment = self.decoder_segment(b1, b2, b3, b4, b5, b6)
        if self.out_features:
            return output_segment, b2, b3, b4, b5, b6
        else:
            return output_segment


class ReconstructiveSubNetwork(nn.Cell):
    def __init__(self, in_channel=3, out_channel=3, base_width=128):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channel, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channel=out_channel)

    def construct(self, x):
        b5 = self.encoder(x)
        output = self.decoder(b5)
        return output


class DraemWithLossCell(nn.Cell):
    def __init__(self, re_net, dis_net):
        super(DraemWithLossCell, self).__init__()
        self.re_net = re_net
        self.dis_net = dis_net
        self.focal_loss = MindFocalLoss()
        self.l2_loss = nn.MSELoss()
        self.ssim_loss = MindSSIM()
        self.cat = ops.Concat(1)

    def construct(self, image, augmented_image, anomaly_mask):
        gray_rec = self.re_net(augmented_image)
        join_in = self.cat([gray_rec, augmented_image])

        out_mask = self.dis_net(join_in)
        out_mask_sm = nn.Softmax(1)(out_mask)

        l2_loss = self.l2_loss(gray_rec, image).mean()
        ssim_loss = self.ssim_loss(gray_rec, image).mean()

        segment_loss = self.focal_loss(out_mask_sm, anomaly_mask).mean()
        loss = l2_loss + ssim_loss + segment_loss
        return loss
