# [Data Efficient Stagewise Knowledge Distillation](https://arxiv.org/abs/1911.06786)

![Stagewise Training Procedure](/image_classification/figures/training_proc.png)

## Table of Contents
- [Data Efficient Stagewise Knowledge Distillation](#data-efficient-stagewise-knowledge-distillation)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Image Classification](#image-classification)
    - [Introduction](#introduction)
    - [Preparation](#preparation)
    - [Experiments](#experiments)
      - [No Teacher](#no-teacher)
      - [Traditional KD (FitNets)](#traditional-kd-fitnets)
      - [FSP KD](#fsp-kd)
      - [Attention Transfer KD](#attention-transfer-kd)
      - [Hinton KD](#hinton-kd)
      - [Simultaneous KD (Proposed Baseline)](#simultaneous-kd-proposed-baseline)
      - [Stagewise KD (Proposed Method)](#stagewise-kd-proposed-method)
  - [Semantic Segmentation](#semantic-segmentation)
    - [Introduction](#introduction-1)
    - [Preparation](#preparation-1)
    - [Experiments](#experiments-1)
      - [No Teacher](#no-teacher-1)
      - [Traditional KD (FitNets)](#traditional-kd-fitnets-1)
      - [Simultaneous KD (Proposed Baseline)](#simultaneous-kd-proposed-baseline-1)
      - [Stagewise KD (Proposed Method)](#stagewise-kd-proposed-method-1)
  - [Citation](#citation)

This repository presents the code implementation for [Stagewise Knowledge Distillation](https://arxiv.org/abs/1911.06786), a technique for improving knowledge transfer between a teacher model and student model.

## Requirements
- Install the dependencies using `conda` with the `requirements.yml` file
    ```
    conda env create -f environment.yml
    ```
- Setup the `stagewise-knowledge-distillation` package itself
    ```
    pip install -e .
    ```
- Apart from the above mentioned dependencies, it is recommended to have an Nvidia GPU (CUDA compatible) with at least 8 GB of video memory (most of the experiments will work with 6 GB also). However, the code works with CPU only machines as well.

## Image Classification
### Introduction
In this work, [ResNet](https://arxiv.org/abs/1512.03385) architectures are used. Particularly, we used ResNet10, 14, 18, 20 and 26 as student networks and ResNet34 as the teacher network. The datasets used are [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), [Imagenette](https://github.com/fastai/imagenette) and [Imagewoof](https://github.com/fastai/imagenette). Note that Imagenette and Imagewoof are subsets of [ImageNet](http://www.image-net.org/).

### Preparation
- Before any experiments, you need to download the data and saved weights of teacher model to appropriate locations. 
- The following script
    - downloads the datasets
    - saves 10%, 20%, 30% and 40% splits of each dataset separately
    - downloads teacher model weights for all 3 datasets

    ```
    # assuming you are in the root folder of the repository
    cd image_classification/scripts
    bash setup.sh
    ```

### Experiments
For detailed information on the various experiments, refer to the paper. In all the image classification experiments, the following common training arguments are listed with the possible values they can take:
- dataset (`-d`) : imagenette, imagewoof, cifar10
- model (`-m`) : resnet10, resnet14, resnet18, resnet20, resnet26, resnet34
- number of epochs (`-e`) : Integer is required
- percentage of dataset (`-p`) : 10, 20, 30, 40 (don't use this argument at all for full dataset experiments)
- random seed (`-s`) : Give any random seed (for reproducibility purposes)
- gpu (`-g`) : Don't use unless training on CPU (in which case, use `-g 'cpu'` as the argument). In case of multi-GPU systems, run `CUDA_VISIBLE_DEVICES=id` in the terminal before the experiment, where `id` is the ID of your GPU according to `nvidia-smi` output.
- Comet ML API key (`-a`) *(optional)* : If you want to use [Comet ML](https://www.comet.ml) for tracking your experiments, then either put your API key as the argument or make it the default argument in the `arguments.py` file. Otherwise, no need of using this argument.
- Comet ML workspace (`-w`) *(optional)* : If you want to use [Comet ML](https://www.comet.ml) for tracking your experiments, then either put your workspace name as the argument or make it the default argument in the `arguments.py` file. Otherwise, no need of using this argument.

In the following subsections, example commands for training are given for one experiment each.
#### No Teacher
Full Imagenette dataset, ResNet10
```
python3 no_teacher.py -d imagenette -m resnet10 -e 100 -s 0
```

#### Traditional KD ([FitNets](https://arxiv.org/abs/1412.6550))
20% Imagewoof dataset, ResNet18
```
python3 traditional_kd.py -d imagewoof -m resnet18 -p 20 -e 100 -s 0
```

#### [FSP KD](https://ieeexplore.ieee.org/document/8100237)
30% CIFAR10 dataset, ResNet14
```
python3 fsp_kd.py -d cifar10 -m resnet14 -p 30 -e 100 -s 0
```

#### [Attention Transfer KD](https://openreview.net/forum?id=Sks9_ajex)
10% Imagewoof dataset, ResNet26
```
python3 attention_transfer_kd.py -d imagewoof -m resnet26 -p 10 -e 100 -s 0
```

#### [Hinton KD](https://arxiv.org/abs/1503.02531)
Full CIFAR10 dataset, ResNet14
```
python3 hinton_kd.py -d cifar10 -m resnet14 -e 100 -s 0
```

#### Simultaneous KD (Proposed Baseline)
40% Imagenette dataset, ResNet20
```
python3 simultaneous_kd.py -d imagenette -m resnet20 -p 40 -e 100 -s 0
```

#### Stagewise KD (Proposed Method)
Full CIFAR10 dataset, ResNet10
```
python3 stagewise_kd.py -d cifar10 -m resnet10 -e 100 -s 0
```

## Semantic Segmentation

### Introduction
In this work, [ResNet](https://arxiv.org/abs/1512.03385) backbones are used to construct symmetric [U-Nets](https://arxiv.org/abs/1505.04597) for semantic segmentation. Particularly, we used ResNet10, 14, 18, 20 and 26 as the backbones for student networks and ResNet34 as the backbone for the teacher network. The dataset used is the Cambridge-driving Labeled Video Database ([CamVid](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/)).

### Preparation
- The following script
    - downloads the data (and shifts it to appropriate folder)
    - saves 10%, 20%, 30% and 40% splits of each dataset separately
    - downloads the pretrained teacher weights in appropriate folder
    ```
    # assuming you are in the root folder of the repository
    cd semantic_segmentation/scripts
    bash setup.sh
    ```

### Experiments
For detailed information on the various experiments, refer to the paper. In all the semantic segmentation experiments, the following common training arguments are listed with the possible values they can take:
- dataset (`-d`) : camvid
- model (`-m`) : resnet10, resnet14, resnet18, resnet20, resnet26, resnet34
- number of epochs (`-e`) : Integer is required
- percentage of dataset (`-p`) : 10, 20, 30, 40 (don't use this argument at all for full dataset experiments)
- random seed (`-s`) : Give any random seed (for reproducibility purposes)
- gpu (`-g`) : Don't use unless training on CPU (in which case, use `-g 'cpu'` as the argument). In case of multi-GPU systems, run `CUDA_VISIBLE_DEVICES=id` in the terminal before the experiment, where `id` is the ID of your GPU according to `nvidia-smi` output.
- Comet ML API key (`-a`) *(optional)* : If you want to use [Comet ML](https://www.comet.ml) for tracking your experiments, then either put your API key as the argument or make it the default argument in the `arguments.py` file. Otherwise, no need of using this argument.
- Comet ML workspace (`-w`) *(optional)* : If you want to use [Comet ML](https://www.comet.ml) for tracking your experiments, then either put your workspace name as the argument or make it the default argument in the `arguments.py` file. Otherwise, no need of using this argument.

Note: Currently, there are no plans for adding Attention Transfer KD and FSP KD experiments for semantic segmentation.

In the following subsections, example commands for training are given for one experiment each.
#### No Teacher
Full CamVid dataset, ResNet10
```
python3 pretrain.py -d camvid -m resnet10 -e 100 -s 0
```

#### Traditional KD ([FitNets](https://arxiv.org/abs/1412.6550))
20% CamVid dataset, ResNet18
```
python3 traditional_kd.py -d camvid -m resnet18 -p 20 -e 100 -s 0
```

#### Simultaneous KD (Proposed Baseline)
40% CamVid dataset, ResNet20
```
python3 simultaneous_kd.py -d camvid -m resnet20 -p 40 -e 100 -s 0
```

#### Stagewise KD (Proposed Method)
10 % CamVid dataset, ResNet10
```
python3 stagewise_kd.py -d camvid -m resnet10 -p 10 -e 100 -s 0
```

## Citation
If you use this code or method in your work, please cite using
```
@misc{kulkarni2019stagewise,
    title={Stagewise Knowledge Distillation},
    author={Akshay Kulkarni and Navid Panchi and Shital Chiddarwar},
    year={2019},
    eprint={1911.06786},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

Built by [Akshay Kulkarni](https://akshayk07.weebly.com/), [Navid Panchi](https://navidpanchi.wixsite.com/home) and [Sharath Chandra Raparthy](https://sharathraparthy.github.io/).
