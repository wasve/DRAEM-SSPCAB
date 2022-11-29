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


from mindspore import save_checkpoint
from mindspore.train.callback import Callback


class SaveCallBack(Callback):
    """EvalCallBack"""
    def __init__(self, re_model, dis_model, run_name, save_path="./"):
        self.re_model = re_model
        self.dis_model = dis_model
        self.run_name = run_name
        self.save_path = save_path

    def epoch_end(self, run_context):
        """epoch_end"""
        print(f"save {run_context.cur_epoch_num} model file")
        save_checkpoint(save_obj=self.re_model, ckpt_file_name=self.save_path + self.run_name + ".ckpt")
        save_checkpoint(save_obj=self.dis_model, ckpt_file_name=self.save_path + self.run_name + "_seg.ckpt")