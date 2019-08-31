# knowledge_distillation
baseline : https://arxiv.org/abs/1412.6550 
(if nothing works out we'll take this as a paper reimplementation of above paper so no harm)


## TODO list
- [x] train teacher network
- [x] pretrain the child network
- [x] try using different sized networks (keep decreasing the size of the network, take it where there is a big difference of accuracy between teacher and 
- [x] train one block at a time of student, then train classifier part on data (works better)
- [x] Use smaller dataset for knowledge distillation 
- [ ] repeat each one five times. 
- [ ] Use bigger resnets as teachers (e.g Teacher: ResNet101, Student: ResNet34, ResNet18)
- [ ] compare with pruning and other such algos 

### Secondary Aims: 
- [ ] Get it to work for Fully Convolutional Networks.

### Long Term Aims:
- [ ] Go for more general algorithm for compression 



## Preliminary Results : 
Note : All accuracies are on validation dataset unless mentioned otherwise. Adam optimizer with learning rate 1e-4 is used everywhere unless otherwise mentioned. The `n` feature maps of teacher model used for training student model are the first `n` feature maps (all of distinct shapes) output by the teacher model.

#### Following results are for the `ResNet34` teacher model and the following student model : 
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

- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained independently | 86.8, 86.6, 87.8, 86.2, 86.2 | 86.72 +- 0.58
| Student model trained using 5 feature maps from teacher and also using data | 87.2, 87.2, 87.0, 87.6, 87.2 | 87.24 +- 0.20|

#### Following results are for the `ResNet34` teacher model and the following student model :
```
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv2(ni, nf) : 
    return conv_layer(ni, nf, stride = 2)

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        
    def forward(self, x): 
        return (x + self.conv1(x))

def conv_and_res(ni, nf): 
    return nn.Sequential(conv2(ni, nf), ResBlock(nf))

def conv_(nf) : 
    return nn.Sequential(conv_layer(nf, nf), ResBlock(nf))
    
net = nn.Sequential(
    conv_layer(3, 64, ks = 7, stride = 2, padding = 3),
    nn.MaxPool2d(3, 2, padding = 1),
    conv_(64),
    conv_and_res(64, 128),
    conv_and_res(128, 256),
    conv_and_res(256, 512),
    AdaptiveConcatPool2d(),
    Flatten(),
    nn.Linear(2 * 512, 256),
    nn.Linear(256, 10)
)
```
- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 89.2, 89.6, 89.6, 90.0, 90.0 | 89.68 +- 0.29 |
| Student model trained using 5 feature maps from teacher and also using data | 90.0, 90.0, 90.8, 90.2, 90.6 | 90.32 +- 0.32 |
| Student model trained using 4 feature maps from teacher and also using data | 88.6, 88.6, 89.6, 89.2, 89.6 | 89.12 +- 0.45 |
| Student model trained using 3 feature maps from teacher and also using data | 88.0, 89.4, 91.4, 89.2, 90.2 | 89.64 +- 1.12 |
| Student model trained stage-wise using feature maps from teacher and classifier part trained using data | N/A | 94.2 |

#### Following results are for the `ResNet34` teacher model and the following student model (one `BasicBlock` less than previous student model) :
```
class Flatten(nn.Module) :
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv2(ni, nf) : 
    return conv_layer(ni, nf, stride = 2)

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf,nf)
        
    def forward(self, x): 
        return (x + self.conv1(x))

def conv_and_res(ni, nf): 
    return nn.Sequential(conv2(ni, nf), ResBlock(nf))

def conv_(nf) : 
    return nn.Sequential(conv_layer(nf, nf), ResBlock(nf))
    
net = nn.Sequential(
    conv_layer(3, 64, ks = 7, stride = 2, padding = 3),
    nn.MaxPool2d(3, 2, padding = 1),
    conv_(64),
    conv_and_res(64, 128),
    conv_and_res(128, 256),
    AdaptiveConcatPool2d(),
    Flatten(),
    nn.Linear(2 * 256, 128),
    nn.Linear(128, 10)
)
```
- Teacher model is pretrained using the same Imagenette dataset (subset of ImageNet) and gets 93.6 % validation accuracy.

| Training method | Model Accuracies (%) (trained 5 times) | Mean Accuracy (%) |
| --------------|------------------------------------| ------------- |
| Student model trained using data only | 90.8, 90.8, 91.2, 90.6, 91.2 | 90.92 +- 0.24 |
| Student model trained using 4 feature maps from teacher and also using data | 90.4, 90.6, 90.6, 90.4, 90.8 | 90.56 +- 0.14 |
| Student model trained using 3 feature maps from teacher and also using data | 90.8, 91.6, 90.2, 90.4, 90.2 | 90.64 +- 0.52 |
