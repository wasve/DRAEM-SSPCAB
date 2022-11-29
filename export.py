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


from mindspore import Tensor, export, load_checkpoint, load_param_into_net
from src.model import ReconstructiveSubNetwork, DiscriminativeSubNetwork
import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import ops


class DreamEval(nn.Cell):
    def __init__(self, rec_model, disc_model):
        super(DreamEval, self).__init__()
        self.re_model = rec_model
        self.dis_model = disc_model
        self.cat = ops.Concat(1)

    def construct(self, x):
        gray_rec = self.re_model(x)
        join_in = self.cat([gray_rec, x])
        out_mask = self.dis_model(join_in)
        out_mask_sm = nn.Softmax(1)(out_mask)
        out_mask_cv = ms.Tensor(out_mask_sm[0, 1, :, :])
        return out_mask_cv


if "__main__" == __name__:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path_re', action='store', type=str, required=True)
    parser.add_argument('--ckpt_path_dis', action='store', type=str, required=True)
    parser.add_argument('--format', action='store', type=str, required=True)
    args_dict = parser.parse_args()

    if args_dict.format != "MINDIR" and args_dict.format != "AIR":
        print("format must be is MINDIR or AIR")
        exit(0)
    re_model = ReconstructiveSubNetwork()
    dis_model = DiscriminativeSubNetwork()
    load_checkpoint(args_dict.ckpt_path_re, net=re_model)
    load_checkpoint(args_dict.ckpt_path_dis, net=dis_model)
    model = DreamEval(re_model, dis_model)
    _input1 = np.random.uniform(0.0, 1.0, size=(1, 3, 256, 256)).astype(np.float32)
    export(model, Tensor(_input1, ms.float32), file_name="DRAEM", file_format=args_dict.format)


