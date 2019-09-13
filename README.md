# knowledge_distillation
baseline : https://arxiv.org/abs/1412.6550
(if nothing works out we'll take this as a paper reimplementation of above paper so no harm)


## TODO list
- [x] train teacher network
- [x] pretrain the child network
- [x] try using different sized networks (keep decreasing the size of the network, take it where there is a big difference of accuracy between teacher and
- [x] train one block at a time of student, then train classifier part on data (works better)
- [x] Use smaller dataset for knowledge distillation
- [x] Use bigger resnets as teachers (done with ResNet50)
- [x] Use smaller dataset for training and test it on bigger dataset (training dataset is 1/4 of the original dataset rest is for testing).
- [ ] If the above step doesn't work out, create the student network by cutting a pretrained network on ImageNet dataset. 
- [x] Repeat experiments using Imagewoof (since it presents a more difficult classification problem compared to Imagenette). 
- [ ] Repeat each experiment 5 times with different random seeds.
- [ ] compare with pruning and other such algos

### Secondary Aims:
- [ ] Get it to work for Unet.

#### Roadmap for Unet:
Need to test two things:
1) - [ ] Smaller encoder and corresponding decoder. (Written in notebooks/unet.py)
2) - [ ] Smaller encoder and default fastai decoder.



### Long Term Aims:
- [ ] Go for more general algorithm for compression

  

## Results using Imagenette :
Note : All accuracies are on validation dataset unless mentioned otherwise. Adam optimizer with learning rate 1e-4 is used everywhere unless otherwise mentioned. 
### ResNet34 Teacher Model :
- Teacher model is pretrained on ImageNet and gets 93.6 % validation accuracy on Imagenette.
#### [Medium-Sized ResNet34-Type Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/medium_model.py) :
- Experiments on subset of training data (1 / 4th of the original training data) also have Test set which is the remaining data (3 / 4th of the original training data). 
- Note that teacher model gets 98.02 % accuracy on that 3 / 4th of original training data. Since the teacher was actually trained on the entire original training data (as Imagenette is a subset of ImageNet), so it is justified that the value is so high.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 89.2, 89.6, 89.6, 90.0, 90.0 | 89.68 +- 0.29 |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 94.2, 93.6, 93.8, 93.8, 94.2 | 93.92 +- 0.24 |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using subset of training data | 91.8 (91.51 test acc) | N/A |
| Student model trained using only subset of training data | 82.4 (77.45 test acc) | N/A |

#### [Small-Sized ResNet34-Type Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/small_model.py) :

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 90.8, 90.8, 91.2, 90.6, 91.2 | 90.92 +- 0.24 |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 91.8, 92.0, 91.8, 89.8, 91.0 | 91.28 +- 0.81 |

#### [Smallest Student Model (having same feature maps as ResNet34, but no skip connections)](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/smallest_model.py) :

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 88.4 | N/A |

### ResNet50 Teacher Model
- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 98.2 % validation accuracy
#### [Small ResNet50-Type Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/large_model.py) :

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 93.4 | N/A |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 98.2 | N/A |

## Results using Imagewoof :
Note : All accuracies are on validation dataset unless mentioned otherwise. Adam optimizer with learning rate 1e-4 is used everywhere unless otherwise mentioned. 
### ResNet34 Teacher Model :
- Teacher model is pretrained on Imagewoof and gets 91.4 % validation accuracy on Imagenette.
#### [Medium-Sized ResNet34-Type Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/medium_model.py) :
- Experiments on subset of training data (1 / 4th of the original training data) also have Test set which is the remaining data (3 / 4th of the original training data). 
- Note that teacher model gets 97.15 % accuracy on that 3 / 4th of original training data. Since the teacher was actually trained on the entire original training data, so it is justified that the value is so high.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 73.8 | N/A |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 87.0 | N/A |
| Student model trained using only subset of training data | 57.6 (54.63 test acc) | N/A |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using subset of training data | 80.2 (77.32 test acc) | N/A |

#### [Small-Sized ResNet34-Type Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/small_model.py) :
| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 74.6 | N/A |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | 76.8 | N/A |
