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

from mindspore import context, TimeMonitor, LossMonitor
import argparse
from src.callback import SaveCallBack
from src.dataset import *
from src.model import *
from mindspore import nn

ms.set_seed(0)


def train(obj_names, args):
    resize_shape = [256, 256]
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device)
    context.set_context(device_id=args.device_id)

    for obj_name in obj_names:
        gamma = 0.2
        run_name = 'DRAEM_test_' + str(args.lr) + '_' + str(args.epochs) + '_bs' + str(args.bs) + "_" + obj_name
        re_model = ReconstructiveSubNetwork(in_channel=3, out_channel=3)
        dis_model = DiscriminativeSubNetwork(in_channel=6, out_channel=2)
        model_loss = DraemWithLossCell(re_model, dis_model)
        # dataset
        dataset, lens = creat_mvtec_draem_train_dataset(root_dir=args.data_path + f"{obj_name}/train/good",
                                                        anomaly_source_path=args.anomaly_source_path + r"images/",
                                                        resize_shape=resize_shape,
                                                        batch_size=args.bs)
        # learning scheduler
        milestone = [int(args.epochs * 0.1 * lens / args.bs), int(args.epochs * 0.5 * lens / args.bs),
                     int(args.epochs * 0.7 * lens / args.bs), int(args.epochs * lens / args.bs)]
        learning_rates = [args.lr, args.lr * gamma, args.lr * gamma ** 2, args.lr * gamma ** 3]
        lr_scheduler = nn.piecewise_constant_lr(milestone, learning_rates)
        opt = nn.Adam(model_loss.trainable_params(), learning_rate=lr_scheduler)
        train_net = nn.TrainOneStepCell(model_loss, opt)
        model_loss.set_train()
        # train
        model = ms.Model(train_net, dataset)
        save_model = SaveCallBack(re_model, dis_model, run_name)
        cb = [TimeMonitor(), LossMonitor(), save_model]
        model.train(epoch=args.epochs, train_dataset=dataset, callbacks=cb, dataset_sink_mode=True)


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument('--device_id', action='store', type=int, default=0, required=False)
    parse.add_argument('--device', type=str, default="Ascend")
    parse.add_argument('--obj_id', action='store', default=-1, type=int, required=True)
    parse.add_argument('--bs', action='store', default=4, type=int, required=True)
    parse.add_argument('--lr', action='store', default=1e-4, type=float, required=True)
    parse.add_argument('--epochs', action='store', default=100, type=int, required=True)
    parse.add_argument('--data_path', action='store', default=r"mvtec_anomaly_detection", type=str, required=True)
    parse.add_argument('--anomaly_source_path', action='store', default=r"dtd/images", type=str, required=True)
    parse.add_argument('--checkpoint_path', action='store', default=r"./checkpoint_path/", type=str, required=True)
    parse.add_argument('--step_log', action='store', default=5, type=int, required=True)

    args_list = parse.parse_args()

    obj_batch = [['capsule'],
                 ['bottle'],
                 ['carpet'],
                 ['leather'],
                 ['pill'],
                 ['transistor'],
                 ['tile'],
                 ['cable'],
                 ['zipper'],
                 ['toothbrush'],
                 ['metal_nut'],
                 ['hazelnut'],
                 ['screw'],
                 ['grid'],
                 ['wood']
                 ]

    if int(args_list.obj_id) == -1:
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
        picked_classes = obj_list
    else:
        picked_classes = obj_batch[int(args_list.obj_id)]

    train(picked_classes, args_list)
    print("---------------------------train over-------------------------------")
