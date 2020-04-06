import comet_ml
import torch
from torch.utils.data import DataLoader

import models
from arguments import get_args
from dataset import get_dataset
from trainer import train_stagewise, unfreeze
from utils import *

args = get_args(desc='Stagewise training using less data of UNet based on ResNet encoder')
hyper_params = {
    "dataset": args.dataset,
    "model": args.model,
    "seed": args.seed,
    "num_classes": 12,
    "batch_size": 8,
    "num_epochs": args.epoch,
    "learning_rate": 1e-4,
    "stage": 0,
    "perc": str(args.percentage)
}

torch.manual_seed(hyper_params['seed'])
if args.gpu != 'cpu':
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(hyper_params['seed'])

train_dataset, valid_dataset, num_classes = get_dataset(args.dataset, args.percentage)
hyper_params['num_classes'] = num_classes

trainloader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True, drop_last=True)
valloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

student = models.unet.Unet(hyper_params['model'], classes=num_classes, encoder_weights=None).to(args.gpu)
teacher = models.unet.Unet('resnet34', classes=num_classes, encoder_weights=None).to(args.gpu)
teacher.load_state_dict(torch.load('../saved_models/camvid/resnet34/pretrained_0.pt', map_location=args.gpu))
# Freeze the teacher model
teacher = unfreeze(teacher, 30)

sf_student, sf_teacher = get_features(student, teacher)

train_stagewise(hyper_params, teacher, student, sf_teacher, sf_student, trainloader, valloader, args)
