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
    echo "|| Please run the script as: "
    echo "|| bash run_all_mvtec.sh [DATASET_PATH] [ANOMAL_PATH] [CATEGORY] [DEVICE_ID]"
    echo "|| For example: bash ./run_all_mvtec.sh ../data/mvtec/ ../data/dtd/ 1 1"
    echo "|| It is better to use the absolute path."
    echo "||  CATEGORY SELECT "
    echo "||  0  : capsule "
    echo "||  1  : bottle "
    echo "||  2  : carpet "
    echo "||  3  : leather "
    echo "||  4  : pill "
    echo "||  5  : transistor "
    echo "||  6  : tile "
    echo "||  7  : cable "
    echo "||  8  : zipper "
    echo "||  9  : toothbrush "
    echo "||  10 : metal_nut "
    echo "||  11 : hazelnut "
    echo "||  12 : screw "
    echo "||  13 : grid "
    echo "||  14 : wood "
    echo "||  -1 : all "
    echo "=============================================================================================================="
exit 1
fi
set -e

DATA_PATH=$1
ANOMAL_PATH=$2
CATEGORY=$3
DEVICE_ID=$4


python ../train.py \
--device_id $DEVICE_ID \
--obj_id $CATEGORY \
--lr 0.0001 \
--epochs 700 \
--data_path $DATA_PATH \
--anomaly_source_path $ANOMAL_PATH \
--checkpoint_path  ./ \
--step_log 5 \
> train.log 2>&1


if [ $? -eq 0 ];then
    echo "[INFO] training success"
else
    echo "[ERROR] training failed"
    exit 2
fi
echo "[INFO] finish"
cd ../