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


import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.ops import operations as P


class SELayer(nn.Cell):
    def __init__(self, num_channels, reduction_ratio=8):
        super(SELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        print(num_channels_reduced)
        self.reduction_reduced = reduction_ratio
        self.fc1 = nn.Dense(num_channels, num_channels_reduced, has_bias=True)
        self.fc2 = nn.Dense(num_channels_reduced, num_channels, has_bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def construct(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.shape
        squeeze_tensor = input_tensor.view((batch_size, num_channels, H * W)).mean(axis=2)

        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.shape
        fc_out_2 = fc_out_2.view((a, b, 1, 1))
        out_tensor = ops.Mul()(input_tensor, fc_out_2)
        return out_tensor


#  SSPCAB Implementation
class SSPCAB(nn.Cell):
    def __init__(self, channels, kernel_dim=1, dilation=1, reduction_ratio=8):
        super(SSPCAB, self).__init__()
        self.pad = kernel_dim + dilation
        self.border_input = kernel_dim + 2 * dilation + 1
        self.relu = nn.ReLU()
        self.se = SELayer(channels, reduction_ratio)
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim, has_bias=True)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim, has_bias=True)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim, has_bias=True)
        self.conv4 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_dim, has_bias=True)
        self.pad_list = ((0, 0), (0, 0), (2, 2), (2, 2))
        self.pad_f = P.Pad(paddings=self.pad_list)

    def construct(self, x):
        x = self.pad_f(x)
        x1 = self.conv1(x[:, :, :-self.border_input, :-self.border_input])
        x2 = self.conv2(x[:, :, self.border_input:, :-self.border_input])
        x3 = self.conv3(x[:, :, :-self.border_input, self.border_input:])
        x4 = self.conv4(x[:, :, self.border_input:, self.border_input:])
        x = self.relu(x1 + x2 + x3 + x4)
        x = self.se(x)
        return x
