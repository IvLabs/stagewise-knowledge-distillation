# Stagewise Knowledge Distillation

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
