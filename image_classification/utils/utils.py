import os
import numpy as np
import torch
from fastai.vision import *
from image_classification.models.custom_resnet import *


class SaveFeatures:
    def __init__(self, m):
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp):
        self.features = outp
    def remove(self):
        self.handle.remove()


def get_savename(hyper_params, experiment):
    assert experiment in ['stagewise-kd', 'traditional-kd', 'simultaneous-kd', 'no-teacher', 'stagewise-classifier']
    less = 'full_data'
    if hyper_params['percentage'] is not None:
        less = 'less_data' + str(hyper_params['percentage'])

    if experiment == 'stagewise-kd':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/stagewise-kd/' + str(hyper_params['model']) + '_stage' + str(hyper_params['stage'])
    
    elif experiment == 'traditional-kd':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/traditional-kd/' + str(hyper_params['model']) + '_stage' + str(hyper_params['stage'])

    elif experiment == 'simultaneous-kd':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/simultaneous-kd/' + str(hyper_params['model'])

    elif experiment == 'stagewise-classifier':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/stagewise-kd/' + str(hyper_params['model']) + '_classifier'
        
    elif experiment == 'no_teacher':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/no-teacher/' + str(hyper_params['model']) + '_classifier'
    
    os.makedirs(savename, exist_ok=True)
    return savename + '/model' + str(hyper_params['seed']) + '.pt'


def get_model(model_name, dataset, data, teach=False):
    if teach:
        load_name = dataset
        if dataset == 'cifar10' : 
            load_name = dataset[ : -2]
        else:
            load_name = dataset + '2'
        teacher = cnn_learner(data, models.resnet34, metrics=accuracy, pretrained=False)
        teacher = teacher.load(os.path.expanduser("~") + '/.fastai/data/' + load_name + '/models/resnet34_' + load_name + '_bs64')
        teacher.freeze()
    
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
    
    if teach:
        return teacher, net
    else:
        return net  


def get_accuracy(dataloader, Net):
    total = 0
    correct = 0
    Net.eval()
    for i, (images, labels) in enumerate(dataloader):
        images = torch.autograd.Variable(images).float()
        labels = torch.autograd.Variable(labels).float()
        
        if torch.cuda.is_available() : 
            images = images.cuda()
            labels = labels.cuda()

        outputs = Net.forward(images)
        outputs = F.log_softmax(outputs, dim = 1)

        _, pred_ind = torch.max(outputs, 1)
        
        total += labels.size(0)
        correct += (pred_ind == labels).sum().item()

    return (correct / total)


def check(model_name, dataset) :
    if dataset == 'imagenette' : 
        path = untar_data(URLs.IMAGENETTE)
    elif dataset == 'cifar10' : 
        path = untar_data(URLs.CIFAR)
    elif dataset == 'imagewoof' : 
        path = untar_data(URLs.IMAGEWOOF)
    
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

    data = ImageDataBunch.from_folder(path, train = 'train', valid = val, bs = 64, size = sz, ds_tfms = tfms).normalize(stats)
    
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
    savename = '../saved_models/' + dataset + '/' + model_name + '_classifier/model0.pt'
    net.load_state_dict(torch.load(savename, map_location = 'cpu'))
    net.cuda()
#     print('stagewise : ', _get_accuracy(data.valid_dl, net))
    stagewise_acc = _get_accuracy(data.valid_dl, net)
    
    savename = '../saved_models/' + dataset + '/' + model_name + '_no_teacher/model0.pt'
    net.load_state_dict(torch.load(savename, map_location = 'cpu'))
    net.cuda()
#     print('no_teacher : ', _get_accuracy(data.valid_dl, net))
    noteacher_acc = _get_accuracy(data.valid_dl, net)
    
    return noteacher_acc, stagewise_acc

def check_ld(model_name, dataset) :
    if dataset == 'imagenette' : 
        path = untar_data(URLs.IMAGENETTE)
    elif dataset == 'cifar10' : 
        path = untar_data(URLs.CIFAR)
    elif dataset == 'imagewoof' : 
        path = untar_data(URLs.IMAGEWOOF)
    
    new_path = path/'new'
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
    savename = '../saved_models/' + dataset + '/less_data/' + model_name + '_classifier/model0.pt'
    net.load_state_dict(torch.load(savename, map_location = 'cpu'))
    net.cuda()
#     print('stagewise : ', _get_accuracy(data.valid_dl, net))
    ld_stagewise_acc = _get_accuracy(data.valid_dl, net)
    
    savename = '../saved_models/' + dataset + '/less_data/' + model_name + '_no_teacher/model0.pt'
    net.load_state_dict(torch.load(savename, map_location = 'cpu'))
    net.cuda()
#     print('no_teacher : ', _get_accuracy(data.valid_dl, net))
    ld_noteacher_acc = _get_accuracy(data.valid_dl, net)
    
    return ld_noteacher_acc, ld_stagewise_acc

def check_simultaneous(model_name, dataset) :
    if dataset == 'imagenette' : 
        path = untar_data(URLs.IMAGENETTE)
    elif dataset == 'cifar10' : 
        path = untar_data(URLs.CIFAR)
    elif dataset == 'imagewoof' : 
        path = untar_data(URLs.IMAGEWOOF)
    
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

    data = ImageDataBunch.from_folder(path, train = 'train', valid = val, bs = 64, size = sz, ds_tfms = tfms).normalize(stats)
    
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
    savename = '../saved_models/' + dataset + '/simultaneous/' + model_name + '_classifier/model0.pt'
    net.load_state_dict(torch.load(savename, map_location = 'cpu'))
    net.cuda()
#     print('stagewise : ', _get_accuracy(data.valid_dl, net))
    stagewise_acc = _get_accuracy(data.valid_dl, net)
        
    return stagewise_acc

def check_teacher(model_name, dataset) :
    if dataset == 'imagenette' : 
        path = untar_data(URLs.IMAGENETTE)
    elif dataset == 'cifar10' : 
        path = untar_data(URLs.CIFAR)
    elif dataset == 'imagewoof' : 
        path = untar_data(URLs.IMAGEWOOF)
    
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

    data = ImageDataBunch.from_folder(path, train = 'train', valid = val, bs = 64, size = sz, ds_tfms = tfms).normalize(stats)

    if model_name == 'resnet34' :
        learn = cnn_learner(data, models.resnet34, metrics = accuracy)
        learn = learn.load('resnet34_' + load_name + '_bs64')
        learn.freeze()
        net = learn.model
    
    net = net.cuda()
    return(_get_accuracy(data.valid_dl, net))
