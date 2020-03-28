from fastai.vision import *

class Flatten(nn.Module) :
    def forward(self, input) :
        return input.view(input.size(0), -1)

def conv2(ni, nf) : 
    return conv_layer(ni, nf, stride = 2)

class ResBlock(nn.Module) :
    def __init__(self, nf) :
        super().__init__()
        self.conv1 = conv_layer(nf, nf)
        
    def forward(self, x) : 
        return (x + self.conv1(x))

def conv_and_res(ni, nf) : 
    return nn.Sequential(conv2(ni, nf), ResBlock(nf))

def conv_(nf) : 
    return nn.Sequential(conv_layer(nf, nf), ResBlock(nf))

class SaveFeatures :
    def __init__(self, m) : 
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp) : 
        self.features = outp
    def remove(self) :
        self.handle.remove()