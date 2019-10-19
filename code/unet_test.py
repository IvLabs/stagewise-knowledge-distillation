from fastai.vision import *
from fastai.callbacks.hooks import *
from fastai.utils.mem import *
from models.unet import get_unet
from models.custom_resnet import *

from torchsummary import summary

resnet = resnet14

print(get_unet(resnet, pretrained=False, n_classes=32))