import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
import matplotlib.pyplot as plt
from fastai.vision import *
import torch
from torchsummary import summary
torch.cuda.set_device(1)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

from models.custom_resnet import *
from utils import _get_accuracy

def check_ldp(model_name, dataset, perc) :
    if dataset == 'imagenette' : 
        path = untar_data(URLs.IMAGENETTE)
    elif dataset == 'cifar10' : 
        path = untar_data(URLs.CIFAR)
    elif dataset == 'imagewoof' : 
        path = untar_data(URLs.IMAGEWOOF)
    
    new_path = path/('new' + str(perc))
    val = 'val'
    sz = 224
    stats = imagenet_stats

    tfms = get_transforms(do_flip=False)
    load_name = dataset
    if dataset == 'cifar10' : 
        val = 'test'
        sz = 32
        stats = cifar_stats
        load_name = dataset[ : -2]

    data = ImageDataBunch.from_folder(new_path, train = 'train', valid = 'val', test = 'test', bs = 64, size = sz, ds_tfms = tfms).normalize(stats)
    
    if model_name == 'resnet10' :
        net = resnet10(pretrained = False, progress = False)
    elif model_name == 'resnet14' : 
        net = resnet14(pretrained = False, progress = False)
    elif model_name == 'resnet18' :
        net = resnet18(pretrained = False, progress = False)
    elif model_name == 'resnet20' :
        net = resnet20(pretrained = False, progress = False)
    elif model_name == 'resnet26' :
        net = resnet26(pretrained = False, progress = False)
    savename = '../saved_models/' + dataset + '/less_data' + str(perc) + '/' + model_name + '_classifier/model0.pt'
    net.load_state_dict(torch.load(savename, map_location = 'cpu'))
    net.cuda()

    ld_stagewise_acc = _get_accuracy(data.valid_dl, net)
        
    return ld_stagewise_acc

# print(check_ldp('resnet10', 'imagenette', 20))
# print(check_ldp('resnet14', 'imagenette', 20))
# print(check_ldp('resnet18', 'imagenette', 20))
# print(check_ldp('resnet20', 'imagenette', 20))
# print(check_ldp('resnet26', 'imagenette', 20))

print(check_ldp('resnet10', 'imagewoof', 20))
print(check_ldp('resnet14', 'imagewoof', 20))
print(check_ldp('resnet18', 'imagewoof', 20))
print(check_ldp('resnet20', 'imagewoof', 20))
print(check_ldp('resnet26', 'imagewoof', 20))