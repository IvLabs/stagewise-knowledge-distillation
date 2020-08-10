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

args = get_args(description='Hinton KD', mode='train')
expt = 'hinton-kd'

torch.manual_seed(args.seed)
if args.gpu != 'cpu':
    args.gpu = int(args.gpu)
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

hyper_params = {
    "dataset": args.dataset,
    "model": args.model,
    "num_classes": 10,
    "batch_size": 64,
    "num_epochs": args.epoch,
    "learning_rate": 1e-4,
    "momentum": 0.9,
    "seed": args.seed,
    "percentage":args.percentage,
    "gpu": args.gpu,
    "temperature" : 20,
    "alpha" : 0.2,
    "weight_decay": 5e-4,
    "stage":0
}

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)


learn, net = get_model(hyper_params['model'], hyper_params['dataset'], data, teach=True)
learn.model, net = learn.model.to(args.gpu), net.to(args.gpu)

teacher = learn.model

sf_student = None
sf_teacher = None

if args.api_key:
    project_name = expt + '-' + hyper_params['model'] + '-' + hyper_params['dataset']
    experiment = Experiment(api_key=args.api_key, project_name=project_name, workspace=args.workspace)
    experiment.log_parameters(hyper_params)

savename = get_savename(hyper_params, experiment=expt)
optimizer = torch.optim.SGD(net.parameters(), lr=hyper_params["learning_rate"], momentum=hyper_params["momentum"], weight_decay=hyper_params["weight_decay"])

loss_function = nn.KLDivLoss(reduction='mean')
loss_function2 = nn.CrossEntropyLoss()
best_val_loss = 100
for epoch in range(hyper_params["num_epochs"]):
    net, train_loss, val_loss, _, best_val_loss = train(net,
                                                        teacher,
                                                        data,
                                                        sf_teacher,
                                                        sf_student,
                                                        loss_function,
                                                        loss_function2,
                                                        optimizer=optimizer,
                                                        hyper_params=hyper_params,
                                                        epoch=epoch,
                                                        savename=savename,
                                                        best_val_acc=best_val_loss,
                                                        expt=expt
                                                        )
    if args.api_key:
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("val_loss", val_loss)

