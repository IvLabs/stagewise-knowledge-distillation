from comet_ml import Experiment

import torch
from torch.utils.data import DataLoader

import models
from arguments import get_args
from dataset import get_dataset
from trainer import pretrain

args = get_args(desc="standalone training for small dataset")
hyper_params = {
    "dataset": args.dataset,
    "model": args.model,
    "seed": args.seed,
    "perc": args.percentage,
    "num_classes": 12,
    "batch_size": 8,
    "num_epochs": args.epoch,
    "learning_rate": 1e-4,
    "repeat": 1
}

torch.manual_seed(hyper_params['seed'])
if args.gpu != 'cpu':
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(hyper_params['seed'])

train_dataset, valid_dataset, num_classes = get_dataset(args.dataset, args.percentage)
hyper_params['num_classes'] = num_classes

trainloader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True, drop_last=True)
valloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)
unet = models.unet.Unet(hyper_params['model'], classes=num_classes, encoder_weights=None).to(args.gpu)

pretrain(hyper_params, unet, valloader, valloader, args)
