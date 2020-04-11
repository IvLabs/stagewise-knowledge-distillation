import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from comet_ml import Experiment
from fastai.vision import *
import torch
import argparse
import os
from image_classification.arguments import get_args
from image_classification.datasets.dataset import get_dataset
from image_classification.utils.utils import *
from image_classification.models.custom_resnet import *
from trainer import *


args = get_args(description='Simultaneous KD', mode='train')

torch.manual_seed(args.seed)
if args.gpu != 'cpu':
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

hyper_params = {
    "dataset": args.dataset,
    "model": args.model,
    "stage": 0,
    "num_classes": 10,
    "batch_size": 64,
    "num_epochs": args.epoch,
    "learning_rate": 1e-4,
    "seed": args.seed,
    "percentage":args.percentage,
    "gpu": args.gpu
}

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)

learn, net = get_model(hyper_params['model'], hyper_params['dataset'], data, teach=True)
learn.model, net = learn.model.to(args.gpu), net.to(args.gpu)

# saving outputs of all Basic Blocks
teacher = learn.model
# for all 5 feature maps
sf_teacher = [SaveFeatures(m) for m in [teacher[0][2], teacher[0][4], teacher[0][5], teacher[0][6], teacher[0][7]]]
sf_student = [SaveFeatures(m) for m in [net.relu2, net.layer1, net.layer2, net.layer3, net.layer4]]

project_name = 'simultaneous-kd-' + hyper_params['model'] + '-' + hyper_params['dataset']
experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name = project_name, workspace="akshaykvnit")
experiment.log_parameters(hyper_params)

optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params["learning_rate"])
loss_function2 = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()
savename = get_savename(hyper_params, experiment='simultaneous-kd')
best_val_acc = 0
for epoch in range(hyper_params['num_epochs']):
    student, train_loss, val_loss, val_acc, best_val_acc = train(
                                                                net,
                                                                teacher,
                                                                data,
                                                                sf_teacher,
                                                                sf_student,
                                                                loss_function,
                                                                loss_function2,
                                                                optimizer,
                                                                hyper_params,
                                                                epoch,
                                                                savename,
                                                                best_val_acc
                                                                )
    experiment.log_metric("train_loss", train_loss)
    experiment.log_metric("val_loss", val_loss)
    experiment.log_metric("val_acc", val_acc * 100)
