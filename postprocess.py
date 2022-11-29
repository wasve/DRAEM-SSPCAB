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
from math import log
from mindspore import nn
import mindspore as ms
import mindspore.ops as ops
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from src.dataset import creat_mvtec_draem_eval_dataset

parser = argparse.ArgumentParser(description='postprocess')

parser.add_argument('--result_dir', type=str, default='')
parser.add_argument('--data_dir', type=str, default='')
parser.add_argument('--category', type=str, default='screw')
parser.add_argument('--device', type=str, default='CPU')
parser.add_argument('--device_id', type=int, default=0)
args = parser.parse_args()


if __name__ == '__main__':
    ms.context.set_context(device_target=args.device)
    ms.context.set_context(device_id=args.device_id)
    img_dim = 256
    ds_test, size = creat_mvtec_draem_eval_dataset(os.path.join(args.data_dir, args.category, "test"))
    dataloader = ds_test.create_dict_iterator(output_numpy=True, num_epochs=1)

    total_pixel_scores = np.zeros((img_dim * img_dim * size))
    total_gt_pixel_scores = np.zeros((img_dim * img_dim * size))
    mask_cnt = 0
    anomaly_score_gt = []
    anomaly_score_prediction = []

    for i_batch, sample_batched in enumerate(dataloader):
        gray_batch = sample_batched["image"]
        is_normal = sample_batched["has_anomaly"][0, 0]
        anomaly_score_gt.append(is_normal)
        true_mask = sample_batched["mask"]
        true_mask_cv = true_mask[0, :, :, :].transpose((1, 2, 0))
        digit = int(log(size, 10)) + 1
        file_name = os.path.join(args.result_dir,  args.category + "_" + str(i_batch).zfill(digit) + "_0" + ".bin")
        out_mask = ms.Tensor(np.fromfile(file_name, np.float32).reshape(1, 2, 256, 256), ms.float32)
        out_mask_sm = nn.Softmax(1)(out_mask)
        out_mask_cv = ms.Tensor(out_mask_sm[0, 1, :, :])
        pool = ops.AvgPool(kernel_size=21, strides=1)
        out_mask_averaged = pool(ops.pad(ms.Tensor(out_mask_sm[:, 1:, :, :]), ((0, 0), (0, 0), (10, 10), (10, 10))))
        image_score = out_mask_averaged.max()
        anomaly_score_prediction.append(image_score)
        flat_true_mask = true_mask_cv.flatten()
        flat_out_mask = out_mask_cv.flatten()
        total_pixel_scores[
        mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask.asnumpy()
        total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
        mask_cnt += 1

    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)

    auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
    ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)

    total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
    total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
    total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]

    auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
    ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)
    print(args.category)
    print("AUC Image:  " + str(auroc))
    print("AP Image:  " + str(ap))
    print("AUC Pixel:  " + str(auroc_pixel))
    print("AP Pixel:  " + str(ap_pixel))
    print("==============================")