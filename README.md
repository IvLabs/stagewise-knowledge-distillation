# knowledge_distillation

## TODO list
- [x] train teacher network
- [ ] pretrain the child network
- [ ] try using different sized networks (keep decreasing the size of the network, take it where there is a big difference of accuracy between teacher and 
student, then do feature distillation 
- [ ] Use smaller dataset for feature distillation 



(if nothing works out we'll take this as a paper reimplementation of https://arxiv.org/abs/1412.6550 so no harm)


## Preliminary Results : 
Note : All accuracies are on validation dataset unless mentioned otherwise.

Following results are for the `ResNet34` teacher model and the following student model
```
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

net = nn.Sequential(
    nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    Flatten(),
    nn.Linear(512 * 7 * 7, 512),
    nn.Linear(512, 10)
)
```

Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets around 94 % validation accuracy.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained independently | 86.8, 86.6, 87.8, 86.2, 86.2 | 86.72 +- 0.58
| Student model trained using 5 feature maps from teacher and also using data | 87.2, 87.2, 87.0, 87.6, 87.2 | 87.24 +- 0.20|