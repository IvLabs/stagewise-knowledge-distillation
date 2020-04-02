import torch
from torch.utils.data import DataLoader

import models
from args import get_args
from dataset import get_dataset
from trainer import unfreeze, train_traditional
from utils import get_features_trad

args = get_args(desc='traditional kd training of UNet based on ResNet encoder')

hyper_params = {
    "model": args.m,
    "seed": args.s,
    "num_classes": 12,
    "batch_size": 8,
    "num_epochs": args.e,
    "learning_rate": 1e-4,
}

torch.cuda.set_device(args.gpu)
torch.manual_seed(args.s)
torch.cuda.manual_seed(args.s)

train_dataset, valid_dataset, num_classes = get_dataset(args.d, args.p)
hyper_params['num_classes'] = num_classes

trainloader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True, drop_last=True)
valloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

student = models.unet.Unet(hyper_params['model'], classes=12, encoder_weights=None).cuda()
teacher = models.unet.Unet('resnet34', classes=12, encoder_weights=None).cuda()
teacher.load_state_dict(torch.load('../saved_models/camvid/resnet34/pretrained_0.pt'))
# Freeze the teacher model
teacher = unfreeze(teacher, 30)

sf_student, sf_teacher = get_features_trad(student, teacher)

train_traditional(hyper_params, teacher, student, sf_teacher, sf_student, trainloader, valloader, args)
