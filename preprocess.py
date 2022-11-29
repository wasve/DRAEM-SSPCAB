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


import os
import argparse
from src.dataset import creat_mvtec_draem_eval_dataset
from math import log

parser = argparse.ArgumentParser(description='preprocesss')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument("--img_dir", type=str, help="")
parser.add_argument('--category', type=str, default='screw')
args = parser.parse_args()


if __name__ == '__main__':
    ds_test, size = creat_mvtec_draem_eval_dataset(os.path.join(args.data_dir, args.category, "test"))
    zero = ""
    for i, data in enumerate(ds_test.create_dict_iterator()):
        img = data['image'].asnumpy()
        digit = int(log(size, 10)) + 1
        # save img
        file_name_img = "data_img" + "_" + args.category + "_" + str(i).zfill(digit) + ".bin"
        file_path = os.path.join(args.img_dir, file_name_img)
        img.tofile(file_path)
