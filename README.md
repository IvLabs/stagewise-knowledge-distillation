# knowledge_distillation
baseline : https://arxiv.org/abs/1412.6550
(if nothing works out we'll take this as a paper reimplementation of above paper so no harm)


## TODO list
- [x] train teacher network
- [x] pretrain the child network
- [x] try using different sized networks (keep decreasing the size of the network, take it where there is a big difference of accuracy between teacher and
- [x] train one block at a time of student, then train classifier part on data (works better)
- [x] Use smaller dataset for knowledge distillation
- [x] repeat each one five times
- [ ] Use bigger resnets as teachers (e.g Teacher: ResNet101, Student: ResNet34, ResNet18)
- [ ] compare with pruning and other such algos

### Secondary Aims:
- [ ] Get it to work for Unet.

#### Roadmap for Unet:
Need to test two things:
1) - [ ] Smaller encoder and corresponding decoder. (Written in notebooks/unet.py)
2) - [ ] Smaller encoder and default fastai decoder.



### Long Term Aims:
- [ ] Go for more general algorithm for compression



## Preliminary Results :
Note : All accuracies are on validation dataset unless mentioned otherwise. Adam optimizer with learning rate 1e-4 is used everywhere unless otherwise mentioned. The `n` feature maps of teacher model used for training student model are the first `n` feature maps (all of distinct shapes) output by the teacher model.

#### Following results are for the `ResNet34` teacher model and [this student model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/archive/models/5fm.py) :

- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained independently | 86.8, 86.6, 87.8, 86.2, 86.2 | 86.72 +- 0.58
| Student model trained using 5 feature maps from teacher and also using data | 87.2, 87.2, 87.0, 87.6, 87.2 | 87.24 +- 0.20|

#### Following results are for the `ResNet34` teacher model and [this student model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/medium_model.py) :

- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 89.2, 89.6, 89.6, 90.0, 90.0 | 89.68 +- 0.29 |
| Student model trained using 5 feature maps from teacher and also using data | 90.0, 90.0, 90.8, 90.2, 90.6 | 90.32 +- 0.32 |
| Student model trained using 4 feature maps from teacher and also using data | 88.6, 88.6, 89.6, 89.2, 89.6 | 89.12 +- 0.45 |
| Student model trained using 3 feature maps from teacher and also using data | 88.0, 89.4, 91.4, 89.2, 90.2 | 89.64 +- 1.12 |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 94.2, 93.6, 93.8, 93.8, 94.2 | 93.92 +- 0.24 |

#### Following results are for the `ResNet34` teacher model and [this student model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/small_model.py) :

- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 90.8, 90.8, 91.2, 90.6, 91.2 | 90.92 +- 0.24 |
| Student model trained using 4 feature maps from teacher and also using data | 90.4, 90.6, 90.6, 90.4, 90.8 | 90.56 +- 0.14 |
| Student model trained using 3 feature maps from teacher and also using data | 90.8, 91.6, 90.2, 90.4, 90.2 | 90.64 +- 0.52 |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 91.8, 92.0, 91.8, 89.8, 91.0 | 91.28 +- 0.81 |

#### Following results are for the `ResNet34` teacher model and [this student model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/smallest_model.py) :

- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 88.4 | N/A |

#### Following results are for the `ResNet50` teacher model and [this student model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/large_model.py) :

- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 98.2 % validation accuracy

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | N/A | N/A |
