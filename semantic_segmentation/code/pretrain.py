import torch
from torch.utils.data import DataLoader

import models
from args import get_args
from dataset import get_dataset
from trainer import pretrain

args = get_args(desc="standalone training for small dataset")
hyper_params = {
    "dataset": args.d,
    "model": args.m,
    "seed": args.s,
    "perc": args.p,
    "num_classes": 12,
    "batch_size": 8,
    "num_epochs": args.e,
    "learning_rate": 1e-4,
    "repeat": 1
}

torch.cuda.set_device(args.gpu)
torch.manual_seed(hyper_params['seed'])
torch.cuda.manual_seed(hyper_params['seed'])

train_dataset, valid_dataset, num_classes = get_dataset(args.d, args.p)
hyper_params['num_classes'] = num_classes

trainloader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True, drop_last=True)
valloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
unet = models.unet.Unet(hyper_params['model'], classes=num_classes, encoder_weights=None).cuda()

pretrain(hyper_params, unet, trainloader, valloader, args)
