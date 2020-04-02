import torch
from torch.utils.data import DataLoader

from args import get_args
from dataset import get_dataset
from trainer import evaluate


args = get_args(desc="args for evaluation", mode='eval')

torch.cuda.set_device(args.gpu)

_, valid_dataset, test_dataset, num_classes = get_dataset(args.d, None, True)

valloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

params = {
    "model": None,
    "seed": args.s,
    'num_classes': num_classes
}

evaluate(valloader, args, params, mode='pretrain') # without teacher training

evaluate(valloader, args, params, mode='classifier') # stagewise training

evaluate(valloader, args, params, mode='simultaneous') # simultanous training

evaluate(valloader, args, params, mode='traditional-kd') # traditional-kd training





