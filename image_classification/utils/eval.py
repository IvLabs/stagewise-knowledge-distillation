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


args = get_args(description='Evaluation Script', mode='eval')

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
    "gpu": args.gpu,
    "experiment": args.experiment
}

data = get_dataset(dataset=hyper_params['dataset'],
                   batch_size=hyper_params['batch_size'],
                   percentage=args.percentage)

model = get_model(hyper_params['model'], hyper_params['dataset'])

savename = get_savename(hyper_params, experiment=hyper_params['experiment'])
model.load_state_dict(torch.load(savename))

if hyper_params['percentage'] is None:
    print(f'{hyper_params['dataset']} - {hyper_params['model']} - Full Data - {get_accuracy(data.valid_dl, model)}')
else:
    print(f'{hyper_params['dataset']} - {hyper_params['model']} - {hyper_params['percentage']}% data - {get_accuracy(data.valid_dl, model)}')