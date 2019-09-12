## Archived Results

Note : All accuracies are on validation dataset unless mentioned otherwise. Adam optimizer with learning rate 1e-4 is used everywhere unless otherwise mentioned. The n feature maps of teacher model used for training student model are the first n feature maps (all of distinct shapes) output by the teacher model.

### ResNet34 Teacher
- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

#### [Student Model](https://github.com/akshaykvnit/knowledge_distillation/blob/master/archive/models/5fm.py)
| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained independently | 86.8, 86.6, 87.8, 86.2, 86.2 | 86.72 +- 0.58
| Student model trained using 5 feature maps from teacher and also using data | 87.2, 87.2, 87.0, 87.6, 87.2 | 87.24 +- 0.20|

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
