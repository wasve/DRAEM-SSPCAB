#!/bin/bash
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

if [ $# != 4 ]
then
    echo "=============================================================================================================="
    echo "Please run the script as: "
    echo "bash run_eval.sh [DATASET_PATH] [CKPT_PATH] [BASE_NAME] [DEVICE_ID] [DEVICE]"
    echo "For example: bash run_eval.sh ../data/mvtec/ ./ DRAEM_test_0.0001_700_bs4 1 Ascend"
    echo "It is better to use the absolute path."
    echo "=============================================================================================================="
exit 1
fi
set -e

DATA_PATH=$1
CKPT_PATH=$2
BASE_NAME=$3
DEVICE_ID=$4
DEVICE=$5


python ../eval.py \
--gpu_id $DEVICE_ID \
--data_path $DATA_PATH \
--base_model_name $BASE_NAME \
--checkpoint_path $CKPT_PATH \
--device $DEVICE \
--device_id $DEVICE_ID \
> eval.log 2>&1

if [ $? -eq 0 ];then
    echo "[INFO] eval success"
else
    echo "[ERROR] eval failed"
    exit 2
fi
echo "[INFO] finish"
cd ../