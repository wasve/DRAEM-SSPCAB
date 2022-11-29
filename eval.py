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


from src.model import *
from src.dataset import *
import mindspore.ops as ops
import mindspore.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import mindspore as ms
import os


def write_results_to_file(run_name, image_auc, pixel_auc, image_ap, pixel_ap):
    if not os.path.exists('./outputs/'):
        os.makedirs('./outputs/')

    fin_str = "img_auc," + run_name
    for i in image_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_auc), 3))
    fin_str += "\n"
    fin_str += "pixel_auc," + run_name
    for i in pixel_auc:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(pixel_auc), 3))
    fin_str += "\n"
    fin_str += "img_ap," + run_name
    for i in image_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(image_ap), 3))
    fin_str += "\n"
    fin_str += "pixel_ap," + run_name
    for i in pixel_ap:
        fin_str += "," + str(np.round(i, 3))
    fin_str += "," + str(np.round(np.mean(pixel_ap), 3))
    fin_str += "\n"
    fin_str += "--------------------------\n"

    with open("./outputs/results.txt", 'a+') as file:
        file.write(fin_str)


def test(obj_names, args):
    ms.context.set_context(device_target=args.device)
    ms.context.set_context(device_id=args.device_id)
    obj_ap_pixel_list = []
    obj_auroc_pixel_list = []
    obj_ap_image_list = []
    obj_auroc_image_list = []
    obj_exit_list = []
    for obj_name in obj_names:
        run_name = args.base_model_name + "_" + obj_name
        if not os.path.exists(args.checkpoint_path + run_name + ".ckpt"):
            continue
        obj_exit_list.append(obj_name)
        mvtec_path_obj = args.data_path + obj_name + "/test"
        img_dim = 256
        model_rec = ReconstructiveSubNetwork(in_channel=3, out_channel=3)
        parameter_rec = ms.load_checkpoint(os.path.join(args.checkpoint_path + run_name + ".ckpt"))
        ms.load_param_into_net(model_rec, parameter_rec)
        model_rec.set_train(False)

        model_dis = DiscriminativeSubNetwork(in_channel=6, out_channel=2)
        parameter_dis = ms.load_checkpoint(os.path.join(args.checkpoint_path + run_name + "_seg.ckpt"))
        ms.load_param_into_net(model_dis, parameter_dis)
        model_dis.set_train(False)

        dataset, size = creat_mvtec_draem_eval_dataset(mvtec_path_obj)
        dataloader = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

        total_pixel_scores = np.zeros((img_dim * img_dim * len(size)))
        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(size)))
        mask_cnt = 0
        anomaly_score_gt = []
        anomaly_score_prediction = []

        for i_batch, sample_batched in enumerate(dataloader):
            gray_batch = sample_batched["image"]

            is_normal = sample_batched["has_anomaly"][0, 0]
            anomaly_score_gt.append(is_normal)
            true_mask = sample_batched["mask"]
            true_mask_cv = true_mask[0, :, :, :].transpose((1, 2, 0))

            gray_rec = model_rec(ms.Tensor(gray_batch))
            joined_in = ops.Concat(1)((gray_rec, ms.Tensor(gray_batch)))

            out_mask = model_dis(joined_in)
            out_mask_sm = nn.Softmax(1)(out_mask)

            out_mask_cv = ms.Tensor(out_mask_sm[0, 1, :, :])
            pool = ops.AvgPool(kernel_size=21, strides=1)
            out_mask_averaged = pool(
                ops.pad(ms.Tensor(out_mask_sm[:, 1:, :, :]), ((0, 0), (0, 0), (10, 10), (10, 10))))
            image_score = out_mask_averaged.max()
            anomaly_score_prediction.append(image_score)
            flat_true_mask = true_mask_cv.flatten()
            flat_out_mask = out_mask_cv.flatten()
            total_pixel_scores[
            mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask.asnumpy()
            total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
            mask_cnt += 1
            print(mask_cnt)

        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt)
        auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
        total_gt_pixel_scores = total_gt_pixel_scores.astype(np.uint8)
        total_gt_pixel_scores = total_gt_pixel_scores[:img_dim * img_dim * mask_cnt]
        total_pixel_scores = total_pixel_scores[:img_dim * img_dim * mask_cnt]
        auroc_pixel = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        ap_pixel = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

        obj_ap_pixel_list.append(ap_pixel)
        obj_auroc_pixel_list.append(auroc_pixel)
        obj_auroc_image_list.append(auroc)
        obj_ap_image_list.append(ap)
        print(obj_name)
        print("AUC Image:  " + str(auroc))
        print("AP Image:  " + str(ap))
        print("AUC Pixel:  " + str(auroc_pixel))
        print("AP Pixel:  " + str(ap_pixel))
        print("==============================")

    print(obj_exit_list)
    print("AUC Image mean:  " + str(np.mean(obj_auroc_image_list)))
    print("AP Image mean:  " + str(np.mean(obj_ap_image_list)))
    print("AUC Pixel mean:  " + str(np.mean(obj_auroc_pixel_list)))
    print("AP Pixel mean:  " + str(np.mean(obj_ap_pixel_list)))

    write_results_to_file(run_name, obj_auroc_image_list, obj_auroc_pixel_list, obj_ap_image_list, obj_ap_pixel_list)


if __name__ == "__main__":
    import argparse
    #
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', action='store', type=int, required=True)
    parser.add_argument('--base_model_name', action='store', type=str, required=True)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--device', type=str, default="Ascend")
    #
    args_dict = parser.parse_args()

    obj_list = ['capsule',
                'bottle',
                'carpet',
                'leather',
                'pill',
                'transistor',
                'tile',
                'cable',
                'zipper',
                'toothbrush',
                'metal_nut',
                'hazelnut',
                'screw',
                'grid',
                'wood'
                ]

    # test(obj_list, "../data/mvtec/", "./", "best_obj_DRAEM_test_1e-05_700_bs4")
    test(obj_list, args_dict)

