# kdsemseg

Note: All experiments are on standard CamVid dataset (unless mentioned otherwise)

### Standalone pretraining from scratch for ResNet based UNets

| Encoder | Train IoU | Val IoU | Test IoU |
| --- | --- | --- | --- |
| ResNet10 | 0.63176 +- 0.01848 | 0.52617 +- 0.00872 | 0.35491 +- 0.00422 |
| ResNet14 | 0.62573 +- 0.01292 | 0.5477 +- 0.00672 | 0.36425 +- 0.00472 |
| ResNet18 | 0.64372 +- 0.0064 | 0.54241 +- 0.00485 | 0.36209 +- 0.00276 |
| ResNet20 | 0.63487 +- 0.00806 | 0.53814 +- 0.01072 | 0.37006 +- 0.0038 |
| ResNet26 | 0.65151 +- 0.01434 | 0.55949 +- 0.01342 | 0.37647 +- 0.00518 |
| ResNet34 | 0.67577 +- 0.00784 | 0.60715 +- 0.00842 | 0.42734 +- 0.006 |

Note : ResNet34 based UNet was trained using ImageNet weights for the encoder as initial weights. Thus, it has higher metric scores and is used for all further experiments as the teacher (unless mentioned otherwise).

### Stagewise training using full training dataset

| Encoder | Train IoU | Val IoU | Test IoU |
| --- | --- | --- | --- |
| ResNet10 | 0.6535 +- 0.00065 | 0.59959 +- 0.00188 | 0.42046 +- 0.00141 |
| ResNet14 | 0.65567 +- 0.00167 | 0.61364 +- 0.00061 | 0.42784 +- 0.00201 |
| ResNet18 | 0.66328 +- 0.00157 | 0.60884 +- 0.00288 | 0.42486 +- 0.00078 |
| ResNet20 | 0.6647 +- 0.00109 | 0.6082 +- 0.00163 | 0.42703 +- 0.00022 |
| ResNet26 | 0.66226 +- 0.0004 | 0.60037 +- 0.00139 | 0.42656 +- 0.00113 |

### Stagewise training for 1/4th of training data

| Encoder | Train IoU | Val IoU | Test IoU |
| --- | --- | --- | --- |
| ResNet10 | 0.62542 +- 0.00096 | 0.55015 +- 0.00484 | 0.38187 +- 0.00239 |
| ResNet14 | 0.61828 +- 0.00688 | 0.56153 +- 0.00226 | 0.3925 +- 0.00252 |
| ResNet18 | 0.63054 +- 0.00495 | 0.56978 +- 0.006 | 0.39678 +- 0.00043 |
| ResNet20 | 0.62635 +- 0.00063 | 0.56548 +- 0.0072 | 0.39863 +- 0.00043 |
| ResNet26 | 0.62772 +- 0.00471 | 0.56626 +- 0.00247 | 0.39943 +- 0.00244 |

### Usage
- Download the data (and shift it to appropriate folder)
```
# assuming you are in the root folder of the repository
cd code
bash get_data.sh
```

- Create folders for saving models (will be useful for evaluation later)
```
# assuming you are in the root folder of the repository
cd code
bash saved_models.sh
```

- Download pretrained teacher weights and move to appropriate folder
```
# assuming you are in the root folder of the repository
cd saved_models
mkdir resnet34
cd resnet34
```
- Download pretrained weight from [this link]()