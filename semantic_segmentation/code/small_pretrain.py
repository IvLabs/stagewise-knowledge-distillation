from comet_ml import Experiment
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import models
import argparse

from dataset import CamVid, Cityscapes
from helper import *
from args import get_args
from trainer import pretrain

torch.cuda.set_device(0)

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

torch.manual_seed(hyper_params['seed'])
torch.cuda.manual_seed(hyper_params['seed'])

if args.d == 'camvid':
    num_classes = 12
    DATA_DIR = '../data/CamVid/'
    train_dataset = CamVid(DATA_DIR, mode='train', p=args.p)
    valid_dataset = CamVid(DATA_DIR, mode='val', p=args.p)

elif args.d == 'cityscapes':
    num_classes = 19
    root = '/home/himanshu/cityscape/'
    root2 = '/home/akshay/cityscape/frac' + str(args.p)
    target_transform = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((480, 640)), encode_segmap(ignore_index=250)])
    tfsm = torchvision.transforms.Compose(
        [torchvision.transforms.Resize((480, 640)), torchvision.transforms.ToTensor()])
    train_dataset = Cityscapes(root=root2,
                               folder='leftImg8bit',
                               split='train',
                               target_type='semantic',
                               mode='fine',
                               transform=tfsm,
                               target_transform=target_transform
                               )
    valid_dataset = Cityscapes(root=root,
                               folder='leftImg8bit',
                               split='val',
                               target_type='semantic',
                               mode='fine',
                               transform=tfsm,
                               target_transform=target_transform
                               )

trainloader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True, drop_last=True)
valloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
unet = models.unet.Unet(hyper_params['model'], classes=num_classes, encoder_weights=None).cuda()

pretrain(hyper_params, unet, trainloader,valloader, p=args.p, dataset=args.d)
