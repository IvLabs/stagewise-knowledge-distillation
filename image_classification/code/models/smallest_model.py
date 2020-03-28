class Flatten(nn.Module) :
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv2(ni, nf) : 
    return conv_layer(ni, nf, stride = 2)
    
net = nn.Sequential(
    conv_layer(3, 64, ks = 7, stride = 2, padding = 3),
    nn.MaxPool2d(3, 2, padding = 1),
    conv2(64, 128),
    conv2(128, 256),
    conv2(256, 512),
    AdaptiveConcatPool2d(),
    Flatten(),
    nn.Linear(2 * 512, 256),
    nn.Linear(256, 10)
)