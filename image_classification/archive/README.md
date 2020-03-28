## Archived README

## TODO
- [x] train teacher network
- [x] pretrain the child network
- [x] try using different sized networks (keep decreasing the size of the network, take it where there is a big difference of accuracy between teacher and
- [x] train one block at a time of student, then train classifier part on data (works better)
- [x] Use smaller dataset for knowledge distillation
- [x] Use bigger resnets as teachers (done with ResNet50)
- [x] Use smaller dataset for training and test it on bigger dataset (training dataset is 1/4 of the original dataset rest is for testing).
- [x] Repeat experiments using Imagewoof (since it presents a more difficult classification problem compared to Imagenette). 
- [ ] Repeat each experiment 5 times with different random seeds.
- [ ] Check for transfer learning
- [ ] compare with pruning and other such algos

### Secondary Aims:
- [ ] Get it to work for Semantic Segmentation using U-Net.

#### Roadmap for Unet:
Need to test two things:
- [ ] Smaller encoder and corresponding decoder. (Written in notebooks/unet.py)
- [ ] Smaller encoder and default fastai decoder.

### Long Term Aims:
- [ ] Go for more general algorithm for compression

**Note** : All accuracies are on validation dataset unless mentioned otherwise. Adam optimizer with learning rate 1e-4 is used everywhere unless otherwise mentioned. 
## Results using Imagenette :

### ResNet34 Teacher Model :
#### Teacher model pre-trained on Imagenette with validation accuracy 99.2 % and stagewise training done using all the data.

| Student Model | Validation Accuracy without Teacher (%) | Validation Accuracy with simultaneous training (%) | Validation Accuracy with stagewise training (%) | Difference between Teacher and Student (for stagewise) (%) |
|:-------------:|:---------------------------------------:|:--------------------------------------------------:|:-----------------------------------------------:|:----------------------------------------------------------:|
|    ResNet10   |                   91.8                  |                        92.2                        |                       97.4                      |                             1.8                            |
|    ResNet14   |                   91.2                  |                        93.2                        |                       98.8                      |                             0.4                            |
|    ResNet18   |                   91.4                  |                        92.4                        |                       98.8                      |                             0.4                            |
|    ResNet20   |                   91.6                  |                        92.4                        |                       98.8                      |                             0.4                            |
|    ResNet26   |                   90.6                  |                        91.8                        |                        99                       |                             0.2                            |

#### Teacher model pre-trained on Imagenette with validation accuracy 99.2 % and stagewise training done using 1/4th of data

| Student Model | Validation Accuracy without Teacher (%) | Validation Accuracy with stagewise training(%) | Difference between Teacher and Student (for stagewise) (%) |
|:-------------:|:---------------------------------------:|:----------------------------------------------:|:----------------------------------------------------------:|
|    ResNet10   |                   84.8                  |                      95.4                      |                             3.8                            |
|    ResNet14   |                    85                   |                       95                       |                             4.2                            |
|    ResNet18   |                   85.4                  |                      95.6                      |                             3.6                            |
|    ResNet20   |                    85                   |                      95.8                      |                             3.4                            |
|    ResNet26   |                   83.2                  |                       96                       |                             3.2                            |

## Results using Imagewoof :

### ResNet34 Teacher Model :
#### Teacher model pre-trained on Imagewoof with validation accuracy 91.4 % and stagewise training done using all the data

| Student Model | Validation Accuracy without Teacher (%) | Validation Accuracy with simultaneous training (%) | Validation Accuracy with stagewise training(%) | Difference between Teacher and Student (for stagewise) (%) |
|:-------------:|:---------------------------------------:|:--------------------------------------------------:|:----------------------------------------------:|:------------------------------------------------------------:|
|    ResNet10   |                   80.2                  |                        79.8                        |                      90.6                      | 0.8                                                        |
|    ResNet14   |                   78.6                  |                        79.6                        |                      92.8                      | -1.4                                                       |
|    ResNet18   |                   79.2                  |                         81                         |                      92.4                      | -1                                                         |
|    ResNet20   |                   79.8                  |                        81.4                        |                       92                       | -0.6                                                       |
|    ResNet26   |                   80.2                  |                        84.2                        |                      93.4                      | -2                                                         |

#### Teacher model pre-trained on Imagewoof with validation accuracy 91.4 % and stagewise training done using 1/4th of data

| Student Model | Validation Accuracy without Teacher | Validation Accuracy with stagewise training(%) | Difference between Teacher and Student (for stagewise) (%) |
|:-------------:|:-----------------------------------:|:----------------------------------------------:|:----------------------------------------------------------:|
|    ResNet10   |                 63.2                |                      85.8                      |                             5.6                            |
|    ResNet14   |                 61.6                |                       89                       |                             2.4                            |
|    ResNet18   |                 60.2                |                       89                       |                             2.4                            |
|    ResNet20   |                  60                 |                      87.6                      |                             3.8                            |
|    ResNet26   |                 58.8                |                      89.8                      |                             1.6                            |

## Results using CIFAR10 :

### ResNet34 Teacher Model :
#### Teacher model pre-trained on CIFAR10 with validation accuracy 87.51 % and stagewise training done using all the data

| Student Model | Validation Accuracy without Teacher (%) | Validation Accuracy with simultaneous training (%) | Validation Accuracy with stagewise training(%) | Difference between Teacher and Student (for stagewise) (%) |
|:-------------:|:---------------------------------------:|:--------------------------------------------------:|:----------------------------------------------:|:------------------------------------------------------------:|
|    ResNet10   |                  77.88                  |                        77.32                       |                      84.75                     | 2.76                                                       |
|    ResNet14   |                   77.5                  |                        75.98                       |                      84.97                     | 2.54                                                       |
|    ResNet18   |                  77.35                  |                        76.47                       |                      85.99                     | 1.52                                                       |
|    ResNet20   |                  78.08                  |                        76.79                       |                      86.46                     | 1.05                                                       |
|    ResNet26   |                   78.3                  |                        76.94                       |                      86.62                     | 0.89                                                       |

#### Teacher model : ResNet34 pre-trained on CIFAR10 with validation accuracy 87.51 % and stagewise training done using 1/4th of data
| Student Model | Validation Accuracy without Teacher | Validation Accuracy with stagewise training(%) | Difference between Teacher and Student (for stagewise) (%) |
|:-------------:|:-----------------------------------:|:----------------------------------------------:|:----------------------------------------------------------:|
|    ResNet10   |                66.35                |                      81.59                     |                            5.92                            |
|    ResNet14   |                64.89                |                      82.55                     |                            4.96                            |
|    ResNet18   |                64.66                |                      83.28                     |                            4.23                            |
|    ResNet20   |                65.19                |                      83.24                     |                            4.27                            |
|    ResNet26   |                64.09                |                      83.64                     |                            3.87                            |

## Models parameters and FLOPs:

| Model | MACs (FLOPs) |  Parameters |
|:--------------:|:------------------------------------:|:-------------:|
| ResNet10 | 896.197M | 5.171M |    
| ResNet14 | 1.359G | 11.072M |  
| ResNet18 | 1.824G | 11.441M |    
| ResNet20 | 2.056G | 12.622M |                                                                                                        
| ResNet26 | 2.752G | 17.712M |                                                
| ResNet34 | 3.679G | 21.550M |  

## Archived Results

Note : All accuracies are on validation dataset unless mentioned otherwise. Adam optimizer with learning rate 1e-4 is used everywhere unless otherwise mentioned. The n feature maps of teacher model used for training student model are the first n feature maps (all of distinct shapes) output by the teacher model.

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

### ResNet34 Teacher
- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

#### [Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/archive/models/5fm.py)
| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained independently | 86.8, 86.6, 87.8, 86.2, 86.2 | 86.72 +- 0.58 |
| Student model trained using 5 feature maps from teacher and also using data | 87.2, 87.2, 87.0, 87.6, 87.2 | 87.24 +- 0.20 |

#### [Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/medium_model.py)
| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using 5 feature maps from teacher and also using data | 90.0, 90.0, 90.8, 90.2, 90.6 | 90.32 +- 0.32 |
| Student model trained using 4 feature maps from teacher and also using data | 88.6, 88.6, 89.6, 89.2, 89.6 | 89.12 +- 0.45 |
| Student model trained using 3 feature maps from teacher and also using data | 88.0, 89.4, 91.4, 89.2, 90.2 | 89.64 +- 1.12 |

#### [Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/code/models/small_model.py)
| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using 4 feature maps from teacher and also using data | 90.4, 90.6, 90.6, 90.4, 90.8 | 90.56 +- 0.14 |
| Student model trained using 3 feature maps from teacher and also using data | 90.8, 91.6, 90.2, 90.4, 90.2 | 90.64 +- 0.52 |
