# 目录

[View English](./README.md)

<!-- TOC -->

- [目录](#目录)
- [DRAEM+SSPCAB概述](#DRAEM+SSPCAB概述)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
    - [脚本说明](#脚本说明)
        - [脚本和样例代码](#脚本和样例代码)
        - [脚本参数](#脚本参数)
        - [预训练模型](#预训练模型)
        - [训练过程](#训练过程)
            - [用法](#用法)
        - [评估过程](#评估过程)
            - [用法](#用法-2)
            - [结果](#结果-1)
        - [导出mindir模型](#导出mindir模型)
        - [推理过程](#推理过程)
            - [用法](#用法-3)
            - [结果](#结果-2)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [训练性能](#训练性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# DRAEM+SSPCAB概述
此模型为无监督学习，通过对好样本手动添加Perlin噪声，成为有缺陷的样本，之后送给网络去训练。重建出输入样本，之后通过鉴别网络，将缺陷处的二值图表现出来.在此网络的倒数第二层的卷积层添加SSPCAB自监督模块模块，来提高模型的整体性能。


# 数据集

使用的训练数据集：[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

- 数据集大小：4.9G，共15个类、5354张图片(尺寸在700x700~1024x1024之间)
    - 训练集：共3629张
    - 测试集：共1725张
- 纹理数据集：<br />
DTD是一个纹理数据库，由5640幅图像组成，根据人类感知的47个术语(类别)列表组织。 每个类别有120张图片。 图像大小在300x300和640x640之间，图像包含至少90%的表面表示类别属性， 这些图片是从谷歌和Flickr收集的。
# 环境要求

- 硬件：昇腾处理器（Ascend/GPU）
    - 使用Ascend/GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend

```shell
# 在单张卡上执行所有的mvtec数据
cd scripts
bash run_all_mvtec.sh [DATASET_PATH] [ANOMAL_PATH] [CATEGORY] [DEVICE_ID]

# 例如
cd scripts
bash ./run_all_mvtec.sh ../data/mvtec/ ../data/dtd/ 1 1 

# 运行评估示例
cd scripts
bash run_eval.sh [DATASET_PATH] [CKPT_PATH] [BASE_NAME] [DEVICE_ID] [DEVICE]

# 例如
cd scripts
bash run_eval.sh ../data/mvtec/ ./ DRAEM_test_0.0001_700_bs4 1
```
- GPU

```shell
# 单机训练运行示例
bash scripts/run_train_gpu.sh [DATASET_PATH] [CKPT_PATH] [BASE_NAME] [DEVICE_ID]

# 运行评估示例
bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [CATEGORY] [DEVICE_ID]
```

## 脚本说明

## 脚本和样例代码

```text
DRAEM+SSPCAB
│  eval.py
│  export.py
│  postprocess.py
│  preprocess.py
│  README.md
│  README_CN.md         // 描述
│  requirements.txt
│  train.py
│
├─ascend310_infer       // 310推理
│  ├─inc
│  │      utils.h
│  │
│  └─src
│          build.sh
│          CMakeList.txt
│          main.cc
│          utils.cc
│
├─scripts
│      download_dataset.sh
│      run_all_mvtec.sh
│      run_eval.sh
│      run_eval_gpu.sh
│      run_infer_310.sh
│      run_train_gpu.sh
│
└─src
    │  dataset.py   // 数据集
    │  loss.py      // 损失函数
    │  model.py     // 模型文件
    │  perlin.py    // 泊林概率函数
    │__sspcab.py    // SSPCAB模块
```

## 脚本参数

```text
train.py和eval.py中主要参数如下：

-- device: 可选值范围为[Ascend, GPU]. 默认Ascend.
-- device_id：用于训练或评估数据集的设备ID。当使用train.sh进行分布式训练时，忽略此参数。
-- checkpoint_path：checkpoint的输出路径。
-- obj_id: 训练对象的索引
-- bs: batch
-- lr: 学习率
-- epochs: 训练的轮数
-- anomaly_source_path：保存推理图片的路径。
-- data_path：训练集路径。
-- step_log: 每几个step打印一次数据
-- base_model_name： 模型的前缀名字
```

## 训练过程

### 用法

- Ascend
```shell
bash scripts/run_all_mvtec.sh [DATASET_PATH] [BACKONE_PATH] [DEVICE_NUM]
```

- GPU
```shell
bash scripts/run_trian_gpu.sh [DATASET_PATH] [BACKONE_PATH] [CATEGORY] [DEVICE_ID]
```

上述shell脚本将在后台运行训练。可以通过`train.log`文件查看结果。

## 评估过程

### 用法

- 在Ascend环境运行时评估

```shell
bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CATEGORY] [DEVICE_ID]
```

- 在GPU环境运行时评估

```shell
bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [CATEGORY] [DEVICE_ID]
```

### 结果

上述python命令将在后台运行，您可以通过./outputs/results.txt文件查看结果。测试数据集的准确性如下：
例如：

| model name                           | img_auc | pixel_auc | img_ap | pixel_ap |
|--------------------------------------|---------|-----------|--------|---|
| DRAEM_test_0.0001_700_bs4_toothbrush | 1.0     | 0.983     | 1.0    |0.609|
| DRAEM_test_0.0001_700_bs4_bottle     | 0.9734  | 0.9864| 0.969  |0.877|
| DRAEM_test_0.0001_700_bs4_capsule    | 0.9734     | 0.9323| 0.9946 |0.4782|
| DRAEM_test_0.0001_700_bs4_leather    | 1     | 0.986     | 0.9866 |0.633|
| DRAEM_test_0.0001_700_bs4_pill       | 0.9809     | 0.983     | 0.996  |0.6808|
| DRAEM_test_0.0001_700_bs4_grid       | 1.0     | 0.983     | 1.0    |0.609|
| DRAEM_test_0.0001_700_bs4_screw      | 1.0     | 0.983     | 1.0    |0.609|

## 导出mindir模型

```python
python export.py --ckpt_path_re [RE_CKPT_PATH] --ckpt_path_dis [DIS_CKPT_PATH] --format [FILE_FORMAT]
```

参数`ckpt_file` 是必需的，`EXPORT_FORMAT` 必须在 ["AIR", "MINDIR"] 中进行选择。

# 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

## 用法

在执行推理之前，需要通过`export.py`导出mindir文件。


```python
python export.py --ckpt_path_re [RE_CKPT_PATH] --ckpt_path_dis [RE_CKPT_PATH] --format [FILE_FORMAT]
```
`ckpt_path_re`，`ckpt_path_dis` 必须填写, `EXPORT_FORMAT` 从 [“AIR”, “MINDIR”]中选择。

```shell
# Ascend310 推理
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE] [DEVICE_ID] [CATEGORY]
```

`DEVICE_TARGET` 可选值范围为：['GPU', 'CPU', 'Ascend']，`NEED_PREPROCESS` 表示数据是否需要预处理，可选值范围为：'y' 或者 'n'，这里直接选择‘y’，`DEVICE_ID` 可选, 默认值为0。

### 结果

推理结果保存在当前路径，可在`acc.log`中看到最终精度结果。

```text
toothbrush
AUC Image:  0.9805555555555555
AP Image:  0.9929775588396278
AUC Pixel:  0.9822477178459982
AP Pixel:  0.6144300312681462
==============================
```

# 模型描述

## 性能

### 训练性能

| 参数          | Ascend                                 | GPU                                                            |
| ------------- |----------------------------------------|----------------------------------------------------------------|
| 模型版本      | DRAEM+SSPACB                           | DRAEM+SSPCAB                                                   |
| 资源          | Ascend 910； CPU： 2.60GHz，192内核；内存，755G | Ubuntu 18.04.6, Tesla V100 1p, CPU 2.90GHz, 64cores, RAM 252GB |
| 上传日期      | 2022-11-25                             | 2022-09-20                                                     |
| MindSpore版本 | 1.8.1                                  | 1.5.0                                                          |
| 数据集        | MVTec AD DTD                           | MVTec AD (zipper) DTD                                          |
| 训练参数      | lr=0.0001, epochs=700                  | lr=0.0001, epochs=700                                          |
| 优化器        | Adam                                   | Adam                                                           |
| 损失函数      | MSELoss    SSIM FocalLOSS              | MSELoss   SSIM  FocalLoss                                      |
| 输出          | 概率                                     | 概率                                                             |
| 损失          | 0.041                                  | 0.04                                                           |
| 速度          | 160 毫秒/步                               | 330 毫秒/步                                                       |
| 总时间        | 1卡：2.5小时                               | 5h                                                             |


# ModelZoo主页  

请浏览官网[主页](https://gitee.com/mindspore/models)。
