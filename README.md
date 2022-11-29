# Content

[查看中文](./README_CN.md)

<!-- TOC -->

- [Content](#content)
- [SSPCAB Description](#stpm-description)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Pretrained Model](#pretrained-model)
    - [Training Process](#training-process)
        - [Usage](#usage)
    - [Evaluation Process](#evaluation-process)
        - [Usage](#usage-2)
        - [Result](#result-1)
    - [Export mindir model](#export-mindir-model)
    - [Inference Process](#inference-process)
        - [Usage](#usage-3)
        - [Result](#result-2)
- [Model Description](#model-description)
    - [Performance](#performance)
- [Description of Random State](#description-of-random-state)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# STPM Model

This model is unsupervised learning. By manually adding Perlin noise to good samples, it becomes defective samples, which are then sent to the network for training. The input sample is reconstructed, and then the binary map of the defect is displayed through the identification network SSPCAB self-monitoring module is added to the convolution layer of the penultimate layer of this network to improve the overall performance of the model.
# Dataset

Dataset used：[MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)

- Dataset size：4.9G，15 classes、5354 images(700x700-1024x1024)
    - Train：3629 images
    - Val：1725 images
- Texture dataset:<br/>
DTD is a texture database, consisting of 5640 images, organized according to the list of 47 terms (categories) perceived by humans. Each category has 120 pictures. The image size is between 300x300 and 640x640. The image contains at least 90% of the surface representation category attributes. These images are collected from Google and Flickr.
# Environment Requirements

- Hardware: Ascend/GPU
    - Prepare hardware environment with Ascend or GPU.
- Framework
    - [MindSpore](https://www.mindspore.cn/install)
- For more information about MindSpore, please check the resources below:
    - [MindSpore tutorial](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

# Quick start

After installing MindSpore through the official website, you can follow the steps below for training and evaluation:

- Ascend

```shell
# single card training
cd scripts
bash run_all_mvtec.sh [DATASET_PATH] [ANOMAL_PATH] [CATEGORY] [DEVICE_ID]
# for example:
cd scripts
bash ./run_all_mvtec.sh ../data/mvtec/ ../data/dtd/ 1 1 

# eval
cd scripts
bash run_eval.sh [DATASET_PATH] [CKPT_PATH] [BASE_NAME] [DEVICE_ID] [DEVICE]

# for example:
cd scripts
bash run_eval.sh ../data/mvtec/ ./ DRAEM_test_0.0001_700_bs4 1
```
- GPU
```shell
# train
bash scripts/run_train_gpu.sh [DATASET_PATH] [CKPT_PATH] [BASE_NAME] [DEVICE_ID]

# eval
bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [CATEGORY] [DEVICE_ID]
```

## Script Description

## Script and Sample Code


```text
DRAEM+SSPCAB
│  eval.py
│  export.py
│  postprocess.py
│  preprocess.py
│  README.md
│  README_CN.md         // Chinese description
│  requirements.txt
│  train.py
│
├─ascend310_infer       // 310 inference
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
    │  dataset.py   // datasets
    │  loss.py      // loss function
    │  model.py     // model file
    │  perlin.py    // perlin 
    │__sspcab.py    // SSPCAB
```

## Script Parameters

```text
main parameters in train.py and eval.py:
The main parameters in train.py and eval.py are as follows:

-- device: The range of optional values is [Ascend, GPU] Default Ascend
-- device_id: Device ID used for training or evaluation dataset.
-- checkpoint_path: The output path of the checkpoint.
-- obj_id: Index of training object
-- bs: Batch
-- lr: Learning rate
-- epochs: Number of training rounds
-- anomaly_source_path: Texture picture path
-- data_path: Training set path.
-- step_log: Print data every few steps
-- base_model_name: Prefix name of the model
```
## Training Process

### Usage

- Ascend

```shell
bash scripts/run_all_mvtec.sh [DATASET_PATH] [BACKONE_PATH] [DEVICE_NUM]
```

- GPU
```shell
bash scripts/run_trian_gpu.sh [DATASET_PATH] [BACKONE_PATH] [CATEGORY] [DEVICE_ID]
```

The above shell script will run the training in the background. The results can be viewed through the `train.log` file.

## Evaluation process

### Usage

- Ascend

```shell
bash scripts/run_eval.sh [DATASET_PATH] [CHECKPOINT_PATH] [CATEGORY] [DEVICE_ID]
```

- GPU

```shell
bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_PATH] [CATEGORY] [DEVICE_ID]
```

### Result

The above python command will run in the background and you can view the results in `./outputs/results.txt` file. The accuracy of the test dataset is as follows:


| model name                           | img_auc | pixel_auc | img_ap | pixel_ap |
|--------------------------------------|---------|-----------|--------|---|
| DRAEM_test_0.0001_700_bs4_toothbrush | 1.0     | 0.983     | 1.0    |0.609|
| DRAEM_test_0.0001_700_bs4_bottle     | 0.9734  | 0.9864| 0.969  |0.877|
| DRAEM_test_0.0001_700_bs4_capsule    | 0.9734     | 0.9323| 0.9946 |0.4782|
| DRAEM_test_0.0001_700_bs4_leather    | 1     | 0.986     | 0.9866 |0.633|
| DRAEM_test_0.0001_700_bs4_pill       | 0.9809     | 0.983     | 0.996  |0.6808|
| DRAEM_test_0.0001_700_bs4_grid       | 1.0     | 0.983     | 1.0    |0.609|
| DRAEM_test_0.0001_700_bs4_screw      | 1.0     | 0.983     | 1.0    |0.609|


## Export mindir model

```python
python export.py --ckpt_path_re [RE_CKPT_PATH] --ckpt_path_dis [RE_CKPT_PATH] --format [FILE_FORMAT]
```

Argument `ckpt_path_re`, `ckpt_path_dis` is required, `EXPORT_FORMAT` choose from ["AIR", "MINDIR"].

# Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

## Usage

Before performing inference, the mindir file needs to be exported via `export.py`.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE] [DEVICE_ID] [CATEGORY]
```

`DEVICE_TARGET` choose from：['GPU', 'CPU', 'Ascend']，`NEED_PREPROCESS` Indicates whether the data needs to be preprocessed. The optional value range is:'y' or 'n'，choose ‘y’，`DEVICE_ID` optional, default is 0.

### Result

The inference result is saved in the current path, and the final accuracy result can be seen in `acc.log`.

```text
toothbrush
AUC Image:  0.9805555555555555
AP Image:  0.9929775588396278
AUC Pixel:  0.9822477178459982
AP Pixel:  0.6144300312681462
==============================
```

# Model Description

## Performance

### Training Performance

| Parameter     | Ascend                                            | GPU                                                            |
| ------------- |---------------------------------------------------|----------------------------------------------------------------|
| Model         | DRAEM+SSPACB                                      | DRAEM+SSPACB                                                   |
| Environment   | Ascend 910; CPU: 2.60GHz, 192 cores; memory, 755G | Ubuntu 18.04.6, Tesla V100 1p, CPU 2.90GHz, 64cores, RAM 252GB |
| Upload Date   | 2022-11-25                                        | 2022-09-20                                                     |
| MindSpore version | 1.8.1                                             | 1.12.0                                                         |
| Dataset       | MVTec AD DTD                                      | MVTec AD DTD                                                   |
| Training parameters | lr=0.0001, epochs=700                             | lr=0.0001, epochs=700                                          |
| Optimizer     | Adam                                              | Adam                                                           |
| Loss func     | MSELoss    SSIM FocalLOSS                         | MSELoss    SSIM FocalLOSS                                      |
| Output        | probability                                       | probability                                                    |
| Loss          | 0.041                                             | 0.041                                                          |
| Speed         | 160 ms/step                                       | 330 ms/step                                                    |
| Total time    | 1 card: 2.5h                                      | 5h                                                             |


# Description of Random State

The initial parameters of the network are all initialized at random.

# ModelZoo Homepage  

Please check the official [homepage](https://gitee.com/mindspore/models).
