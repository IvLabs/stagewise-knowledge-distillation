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


def freeze_student(model, hyper_params, experiment):
    assert experiment in ['stagewise-kd', 'traditional-kd']
    # stage training
    if experiment == 'stagewise-kd' and hyper_params['stage'] != 5:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name[5] == str(hyper_params['stage']) and hyper_params['stage'] != 0:
                param.requires_grad = True
            elif (name[0] == 'b' or name[0] == 'c') and hyper_params['stage'] == 0:
                param.requires_grad = True
    # stagewise classifier training (last stage)
    elif experiment == 'stagewise-kd' and hyper_params['stage'] == 5:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name[0] == 'f':
                param.requires_grad = True
    # traditional-kd first stage
    elif experiment == 'traditional-kd' and hyper_params['stage'] == 0:
        for name, param in model.named_parameters() : 
            param.requires_grad = False
            if (name[0] == 'b' or name[0] == 'c' or name[5] == str(0) or name[5] == str(1) or name[5] == str(2)) : 
                param.requires_grad = True
    # traditional-kd last stage
    elif experiment == 'traditional-kd' and hyper_params['stage'] == 1:
        for name, param in model.named_parameters() : 
            param.requires_grad = False
            if not (name[0] == 'b' or name[0] == 'c' or name[5] == str(0) or name[5] == str(1) or name[5] == str(2)) : 
                param.requires_grad = True

    return model


def get_savename(hyper_params, experiment):
    assert experiment in ['stagewise-kd', 'traditional-kd', 'simultaneous-kd', 'no-teacher']
    less = 'full_data'
    if hyper_params['percentage'] is not None:
        less = 'less_data' + str(hyper_params['percentage'])

    if experiment == 'stagewise-kd':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/stagewise-kd/' + str(hyper_params['model']) + '_stage' + str(hyper_params['stage'])
    
    elif experiment == 'traditional-kd':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/traditional-kd/' + str(hyper_params['model']) + '_stage' + str(hyper_params['stage'])

    elif experiment == 'simultaneous-kd':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/simultaneous-kd/' + str(hyper_params['model'])
        
    elif experiment == 'no-teacher':
        savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + less + '/no-teacher/' + str(hyper_params['model']) + '_classifier'
    
    os.makedirs(savename, exist_ok=True)
    return savename + '/model' + str(hyper_params['seed']) + '.pt'


def get_model(model_name, dataset, data=None, teach=False):
    if teach and data:
        load_name = dataset
        if dataset == 'cifar10' : 
            load_name = dataset[ : -2]
        else:
            load_name = dataset + '2'
        teacher = cnn_learner(data, models.resnet34, metrics=accuracy, pretrained=False)
        teacher = teacher.load(os.path.expanduser("~") + '/.fastai/data/' + load_name + '/models/resnet34_' + load_name + '_bs64')
        teacher.freeze()

    if model_name == 'resnet10' :
        net = resnet10(pretrained=False, progress=False)
    elif model_name == 'resnet14' : 
        net = resnet14(pretrained=False, progress=False)
    elif model_name == 'resnet18' :
        net = resnet18(pretrained=False, progress=False)
    elif model_name == 'resnet20' :
        net = resnet20(pretrained=False, progress=False)
    elif model_name == 'resnet26' :
        net = resnet26(pretrained=False, progress=False)
    elif model_name == 'resnet34' :
        net = resnet34(pretrained=False, progress=False)

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
