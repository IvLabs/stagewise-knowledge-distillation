import os
import numpy as np
import torch
from fastai.vision import *
from image_classification.models import custom_resnet
from pathlib import Path


class SaveFeatures:
    def __init__(self, m):
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp):
        self.features = outp
    def remove(self):
        self.handle.remove()


def get_features(student, teacher, experiment):
    if experiment == 'stagewise-kd' or experiment == 'simultaneous-kd':
        sf_teacher = [SaveFeatures(m) for m in [teacher[0][2], teacher[0][4], teacher[0][5], teacher[0][6], teacher[0][7]]]
        sf_student = [SaveFeatures(m) for m in [student.relu2, student.layer1, student.layer2, student.layer3, student.layer4]]
    elif experiment == 'traditional-kd':
        sf_teacher = [SaveFeatures(m) for m in [teacher[0][5]]]
        sf_student = [SaveFeatures(m) for m in [student.layer2]]
    # features for attention kd and fsp kd are same, but how they are operated on is different
    elif experiment == 'attention-kd' or experiment == 'fsp-kd':
        sf_teacher = [SaveFeatures(m) for m in [teacher[0][4], teacher[0][5], teacher[0][6], teacher[0][7]]]
        sf_student = [SaveFeatures(m) for m in [student.layer1, student.layer2, student.layer3, student.layer4]]
    return sf_student, sf_teacher


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
    assert experiment in ['stagewise-kd', 'traditional-kd', 'simultaneous-kd', 'attention-kd', 'fsp-kd', 'no-teacher']
    
    dsize = 'full_data' if hyper_params['percentage'] is None else f"less_data{str(hyper_params['percentage'])}"

    if experiment in ['stagewise-kd', 'traditional-kd', 'fsp-kd']:
        stage = f"_stage{str(hyper_params['stage'])}"
    elif experiment == "no-teacher":
        stage = f"_classifier"
    else:
        stage = ""

    realpath = Path(os.path.dirname(os.path.realpath(__file__)))
    current_path = Path(os.getcwd())
    parent = realpath.relative_to(current_path)

    if str(parent) == '.':
        new = '../'
    else:
        new = '/'.join(str(parent).split('/')[:-1])
        new = '' if new == '' else new + '/'

    savename = f"{new}saved_models/{hyper_params['dataset']}/{dsize}/{experiment}/{hyper_params['model']}{stage}"
    os.makedirs(savename, exist_ok=True)
    return f"{savename}/model{str(hyper_params['seed'])}.pt"


def get_model(model_name, dataset, data=None, teach=False):
    if teach and data:
        teacher = cnn_learner(data, models.resnet34, metrics=accuracy, pretrained=False)
        teacher = teacher.load(os.path.expanduser("~") + '/.fastai/data/' + dataset + '/models/resnet34_' + dataset + '_bs64')
        teacher.freeze()

    net =  getattr(custom_resnet, model_name)(pretrained=False, progress=False)

    if teach:
        return teacher, net
    else:
        return net


def get_accuracy(dataloader, net):
    total = 0
    correct = 0
    net.eval()
    for i, (images, labels) in enumerate(dataloader):
        images = torch.autograd.Variable(images).float()
        labels = torch.autograd.Variable(labels).float()

        if torch.cuda.is_available() :
            images = images.cuda()
            labels = labels.cuda()

        outputs = net.forward(images)
        outputs = F.log_softmax(outputs, dim = 1)

        _, pred_ind = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (pred_ind == labels).sum().item()

    return (correct / total)


'''
Paying More Attention to Attention: Improving the Performance of Convolutional Neural Networks via Attention Transfer
https://arxiv.org/abs/1612.03928
Code Source : https://github.com/szagoruyko/attention-transfer
'''
def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


'''
A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning
http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
Code Source : https://github.com/AberHu/Knowledge-Distillation-Zoo/blob/master/kd_losses/fsp.py
'''
def fsp_matrix(fm1, fm2):
    if fm1.size(2) > fm2.size(2):
        fm1 = F.adaptive_avg_pool2d(fm1, (fm2.size(2), fm2.size(3)))

    fm1 = fm1.view(fm1.size(0), fm1.size(1), -1)
    fm2 = fm2.view(fm2.size(0), fm2.size(1), -1).transpose(1,2)

    fsp = torch.bmm(fm1, fm2) / fm1.size(2)

    return fsp