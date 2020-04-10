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


args = get_args(description='Stagewise KD', mode='train')

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
if args.gpu != 'cpu':
    learn.model, net = learn.model.to(args.gpu), net.to(args.gpu)

for stage in range(5):
    if hyper_params['stage'] != 0:
        # load previous stage best weights
        filename = get_savename(hyper_params, experiment='stagewise-kd')
        net.load_state_dict(torch.load(filename))
    
    hyper_params['stage'] = stage
    print('stage :', hyper_params['stage'])
    for name, param in net.named_parameters():
        param.requires_grad = False
        if name[5] == str(hyper_params['stage']) and hyper_params['stage'] != 0:
            param.requires_grad = True
        elif (name[0] == 'b' or name[0] == 'c') and hyper_params['stage'] == 0:
            param.requires_grad = True

    # saving outputs of all Basic Blocks
    mdl = learn.model
    sf = [SaveFeatures(m) for m in [mdl[0][2], mdl[0][4], mdl[0][5], mdl[0][6], mdl[0][7]]]
    sf2 = [SaveFeatures(m) for m in [net.relu2, net.layer1, net.layer2, net.layer3, net.layer4]]
    
    project_name = 'stagewise-kd-' + hyper_params['model'] + '-' + hyper_params['dataset']
    experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name = project_name, workspace="akshaykvnit")
    experiment.log_parameters(hyper_params)
    savename = get_savename(hyper_params, experiment='stagewise-kd')
    
    optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params["learning_rate"])
    loss_function = nn.MSELoss()
    best_val_loss = 100
    for epoch in range(hyper_params["num_epochs"]):
        net, train_loss, val_loss, _, best_val_loss = train(net, 
                                                            mdl, 
                                                            data, 
                                                            sf, 
                                                            sf2, 
                                                            loss_function, 
                                                            loss_function2=None, 
                                                            optimizer=optimizer,
                                                            hyper_params=hyper_params,
                                                            epoch=epoch,
                                                            savename=savename,
                                                            best_val_acc=best_val_loss
                                                            )
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("val_loss", val_loss)

# loading previous stage best weights
filename = get_savename(hyper_params, experiment='stagewise-kd')
net.load_state_dict(torch.load(filename))
hyper_params['stage'] = 5

for name, param in net.named_parameters():
    param.requires_grad = False
    if name[0] == 'f':
        param.requires_grad = True
    
project_name = 'stagewise-kd-' + hyper_params['model'] + '-' + hyper_params['dataset']
experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name = project_name, workspace="akshaykvnit")
experiment.log_parameters(hyper_params)

optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params["learning_rate"])
savename = get_savename(hyper_params, experiment='stagewise-kd')
loss_function = nn.CrossEntropyLoss()
best_val_acc = 0

for epoch in range(hyper_params['num_epochs']):
    net, train_loss, val_loss, val_acc, best_val_acc = train(net,
                                                               teacher=None,
                                                               data=data,
                                                               sf_teacher=None,
                                                               sf_student=None,
                                                               loss_function=loss_function,
                                                               loss_function2=None,
                                                               optimizer=optimizer,
                                                               hyper_params=hyper_params,
                                                               epoch=epoch,
                                                               savename=savename,
                                                               best_val_acc=best_val_acc
                                                               )
    experiment.log_metric("train_loss", train_loss)
    experiment.log_metric("val_loss", val_loss)
    experiment.log_metric("val_acc", val_acc * 100)
