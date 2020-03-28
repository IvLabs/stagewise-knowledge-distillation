from comet_ml import Experiment
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import models
import argparse
from helper import *
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description = 'Standalone training of UNet based on ResNet encoder')
parser.add_argument('-m', choices = ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26', 'resnet34'], help = 'Give the encoder name from the choices')
parser.add_argument('-d', choices = ['camvid', 'cityscapes'], help = 'Give the dataset to be used for training from the choices')
parser.add_argument('-e', type = int, help = 'Give number of epochs for training')
parser.add_argument('-s', type = int, help = 'Give the random seed number')
args = parser.parse_args()

hyper_params = {
    "dataset": args.d,
    "model": args.m,
    "seed": args.s,
    "num_classes": 12,
    "batch_size": 8,
    "num_epochs": args.e,
    "learning_rate": 1e-4,
    "repeat": 1
}

torch.manual_seed(hyper_params['seed'])
torch.cuda.manual_seed(hyper_params['seed'])

if args.d == 'camvid' :
    num_classes = 12
    classes = ['sky', 'building', 'pole', 'road', 'pavement', 'tree', 'signsymbol', 'fence', 'car', 'pedestrian', 'bicyclist', 'unlabelled']
    DATA_DIR = '../data/CamVid/'
    x_train_dir = os.path.join(DATA_DIR, 'trainsmall' + str(args.p))
    y_train_dir = os.path.join(DATA_DIR, 'trainsmallannot' + str(args.p))

    x_valid_dir = os.path.join(DATA_DIR, 'val')
    y_valid_dir = os.path.join(DATA_DIR, 'valannot')
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.41189489566336, 0.4251328133025, 0.4326707089857], std = [0.27413549931506, 0.28506257482912, 0.28284674400252])
    ])

    train_dataset = CamVid(
        x_train_dir,
        y_train_dir,
        classes = classes,
        transform = transform
    )

    valid_dataset = CamVid(
        x_valid_dir,
        y_valid_dir,
        classes = classes, 
        transform = transform
    )
    
elif args.d == 'cityscapes' :
    num_classes = 19
    root = '/home/himanshu/cityscape/'
    target_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((480, 640)), encode_segmap(ignore_index = 250)])
    tfsm = torchvision.transforms.Compose([torchvision.transforms.Resize((480, 640)), torchvision.transforms.ToTensor()])
    train_dataset = Cityscapes(root=root, 
                                folder = 'leftImg8bit',
                                split = 'train', 
                                target_type='semantic', 
                                mode='fine', 
                                transform = tfsm, 
                                target_transform = target_transform
                                )
    valid_dataset = Cityscapes(root=root, 
                                folder = 'leftImg8bit',
                                split = 'val', 
                                target_type='semantic', 
                                mode='fine', 
                                transform = tfsm, 
                                target_transform = target_transform
                                )

trainloader = DataLoader(train_dataset, batch_size = hyper_params['batch_size'], shuffle = True, drop_last = True)
valloader = DataLoader(valid_dataset, batch_size = 1, shuffle = False)

project_name = 'pretrain-' + hyper_params['dataset'] + '-' + hyper_params['model']
experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name = project_name, workspace="semseg_kd")
experiment.log_parameters(hyper_params)

unet = models.unet.Unet(hyper_params['model'], classes = num_classes, encoder_weights = None).cuda()

optimizer = torch.optim.Adam(unet.parameters(), lr = hyper_params['learning_rate'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr = 1e-2, steps_per_epoch = len(trainloader), epochs = hyper_params['num_epochs'])
if args.d == 'camvid' :
    criterion = nn.CrossEntropyLoss(ignore_index = 11)
elif args.d == 'cityscapes' : 
    criterion = nn.CrossEntropyLoss(ignore_index = 250)

savename = '../saved_models/' + hyper_params['dataset'] + '/' + hyper_params['model'] + '/pretrained_' + str(hyper_params['seed']) + '.pt'
highest_iou = 0
losses = []
for epoch in range(hyper_params['num_epochs']) :
    unet, highest_iou, train_loss, val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff = train(model = unet, 
                        train_loader = trainloader, 
                        val_loader = valloader,
                        num_classes = num_classes,
                        loss_function = criterion, 
                        optimiser = optimizer, 
                        scheduler = scheduler, 
                        epoch = epoch, 
                        num_epochs = hyper_params['num_epochs'], 
                        savename = savename, 
                        highest_iou = highest_iou
                       )
    experiment.log_metric('train_loss', train_loss)
    experiment.log_metric('val_loss', val_loss)
    experiment.log_metric('avg_iou', avg_iou)
    experiment.log_metric('avg_pixel_acc', avg_pixel_acc)
    experiment.log_metric('avg_dice_coeff', avg_dice_coeff)
