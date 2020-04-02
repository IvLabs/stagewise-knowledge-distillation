import torch
from torch.utils.data import DataLoader

import models
from args import get_args
from dataset import get_dataset
from trainer import train_stagewise, unfreeze
from utils import *

torch.cuda.set_device(0)
args = get_args(desc='Stagewise training using less data of UNet based on ResNet encoder')
hyper_params = {
    "dataset": args.d,
    "model": args.m,
    "seed": args.s,
    "num_classes": 12,
    "batch_size": 8,
    "num_epochs": args.e,
    "learning_rate": 1e-4,
    "stage": 0,
    "perc": str(args.p)
}

torch.manual_seed(args.s)
torch.cuda.manual_seed(args.s)

train_dataset, valid_dataset, num_classes = get_dataset(args.d, args.p)
hyper_params['num_classes'] = num_classes

trainloader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True, drop_last=True)
valloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

student = models.unet.Unet(hyper_params['model'], classes=num_classes, encoder_weights=None).cuda()
teacher = models.unet.Unet('resnet34', classes=num_classes, encoder_weights=None).cuda()
teacher.load_state_dict(torch.load('../saved_models/camvid/resnet34/pretrained_0.pt'))
# Freeze the teacher model
teacher = unfreeze(teacher, 30)

sf_student, sf_teacher = get_features(student, teacher)

train_stagewise(hyper_params, teacher, student, sf_teacher, sf_student, trainloader, valloader, args)
