import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
import matplotlib.pyplot as plt
from fastai.vision import *
import torch
torch.cuda.set_device(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

from models.custom_resnet import *
from utils import *

for dataset in ['imagenette', 'imagewoof', 'cifar10'] :
    imagenette_acc_stagewise = list()
    imagenette_acc_simultaneous = list()
    imagenette_acc_noteacher = list()
    for model in ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26'] :
        noteacher_acc, stagewise_acc = check(model, dataset)
        simul_acc = check_simultaneous(model, dataset)
        imagenette_acc_noteacher.append(noteacher_acc * 100)
        imagenette_acc_stagewise.append(stagewise_acc * 100)
        imagenette_acc_simultaneous.append(simul_acc * 100)

    teacher_acc = check_teacher('resnet34', dataset) * 100

    layers = [10, 14, 18, 20, 26]

    plt.style.use('seaborn-paper')

    fig, ax = plt.subplots()
    title_ = dataset[0].upper() + dataset[1 : ] + ' Accuracy'
    ax.set(xlim = [8, 28], ylim = [50, 100], xlabel = 'Layers in ResNet', ylabel = 'Validation Accuracy', title = title_)
    ax.axhline(teacher_acc, ls = '--', color = 'r', label = 'Teacher Model')
    ax.plot(layers, imagenette_acc_noteacher, 'go-', label = 'Training without Teacher')
    ax.plot(layers, imagenette_acc_simultaneous, 'co-', label = 'Simultaneous Training')
    ax.plot(layers, imagenette_acc_stagewise, 'bo-', label = 'Stagewise Training')
    plt.xticks(layers)
    ax.legend(loc = 'best')
    plt.show()