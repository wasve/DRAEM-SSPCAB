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

if [[ $# != 6 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE] [DEVICE_ID] [CATEGORY]
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'."
exit 1
fi

get_real_path() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}


model=$(get_real_path $1)
dataset_path=$(get_real_path $2)

if [ "$3" == "y" ] || [ "$3" == "n" ]; then
    need_preprocess=$3
else
    echo "weather need preprocess or not, it's value must be in [y, n]"
    exit 1
fi


device_id=$4
device=$5
category=$6


echo "Mindir name: "$model
echo "dataset path: "$dataset_path
echo "need preprocess: "$need_preprocess
echo "device: "$device
echo "device id: "$device_id
echo "category: "$category


function preprocess_data() {
    if [ ! -d img ]; then
        mkdir ./img
    fi
    if [ -d img/$category ]; then
        rm -rf img/$category
    fi
    mkdir ./img/$category
    mkdir ./img/$category/label

    python ../preprocess.py \
    --data_dir $dataset_path \
    --img_dir ./img/$category
}

function compile_app() {
    cd ../ascend_310_infer/src || exit
    bash build.sh &> build.log
}


function infer() {
    cd - || exit
    if [ -d img/$category/result ]; then
        rm -rf img/$category/result
    fi
    mkdir img/$category/result -p

    if [ -d img/$category/time ]; then
        rm -rf img/$category/time
    fi
    mkdir img/$category/time -p

    ../ascend_310_infer/src/out/main \
        --mindir_path=$model \
        --input_path=./img/$category \
        --result_path=./img/$category/result \
        --time_path=./img/$category/time
}

function cal_acc() {
    python ../postprocess.py \
        --result_dir ./img/$category/result/ \
        --data_dir $dataset_path \
        --category $category \
        --device $device \
        --device_id $device_id &> acc_$category.log
}

if [ $need_preprocess == "y" ]; then
   preprocess_data
   if [ $? -ne 0 ]; then
       echo "preprocess dataset failed"
       exit 1
   fi
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi