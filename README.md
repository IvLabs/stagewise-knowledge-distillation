# Stagewise Knowledge Distillation

## Code Implementation for [Stagewise Knowledge Distillation](https://arxiv.org/abs/1911.06786)
This repository presents the code implementation for Stagewise Knowledge Distillation, a technique for improving knowledge transfer between a teacher model and student model.

### Architectures Used
- ResNet10
- ResNet14
- ResNet18
- ResNet20
- ResNet26
- ResNet34

Note: ResNet34 is used as a teacher (being a standard architecture), while others are used as student models.

### Datasets Used
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [Imagenette](https://github.com/fastai/imagenette)
- [Imagewoof](https://github.com/fastai/imagenette)

Note: Imagenette and Imagewoof are subsets of ImageNet.
