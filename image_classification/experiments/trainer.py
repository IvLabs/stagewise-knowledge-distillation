from comet_ml import Experiment
from tqdm import tqdm

from fastai.vision import *

def train(model, data, loss_function, optimizer, hyper_params, savename, best_val_acc):
    loop = tqdm(data.train_dl)
    model.train()
    