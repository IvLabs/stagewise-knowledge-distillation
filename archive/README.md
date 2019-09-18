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
