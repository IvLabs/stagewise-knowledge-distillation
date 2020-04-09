import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from comet_ml import Experiment
from fastai.vision import *
import torch
from models.custom_resnet import *
from utils import _get_accuracy
import argparse
torch.cuda.set_device(0)

parser = argparse.ArgumentParser(description = 'Training of ResNet type model using less data approach')
parser.add_argument('-m', choices = ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26'], help = 'Give the model name from the choices')
parser.add_argument('-d', choices = ['imagenette', 'imagewoof', 'cifar10'], help = 'Give the dataset name from the choices')
parser.add_argument('-p', type = int, help = 'Give percentage of dataset')
parser.add_argument('-e', type = int, help = 'Give number of epochs for training')
parser.add_argument('-s', type = int, help = 'Give random seed')
args = parser.parse_args()

sz = 224
stats = imagenet_stats
num_epochs = args.e
batch_size = 64
# model_name = 'resnet10'
# dataset = 'imagenette'
model_name = args.m
dataset = args.d

if dataset == 'imagenette' : 
    path = untar_data(URLs.IMAGENETTE)
elif dataset == 'cifar10' : 
    path = untar_data(URLs.CIFAR)
elif dataset == 'imagewoof' : 
    path = untar_data(URLs.IMAGEWOOF)

torch.manual_seed(args.s)
torch.cuda.manual_seed(args.s)

hyper_params = {
    "model": model_name, 
    "dataset": dataset,
    "repeated": args.s,
    "num_classes": 10,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": 1e-4,
    "perc": args.p
}

new_path = path/('new' + str(args.p))
batch_size = 64
tfms = get_transforms(do_flip=False)

if hyper_params['dataset'] == 'cifar10' : 
    sz = 32
    stats = cifar_stats

data = ImageDataBunch.from_folder(new_path, train = 'train', valid = 'val', test = 'test', bs = batch_size, size = sz, ds_tfms = tfms).normalize(stats)
    
if model_name == 'resnet10' :
    net = resnet10(pretrained = False, progress = False)
elif model_name == 'resnet14' : 
    net = resnet14(pretrained = False, progress = False)
elif model_name == 'resnet18' :
    net = resnet18(pretrained = False, progress = False)
elif model_name == 'resnet20' :
    net = resnet20(pretrained = False, progress = False)
elif model_name == 'resnet26' :
    net = resnet26(pretrained = False, progress = False)

if torch.cuda.is_available() : 
    net = net.cuda()

project_name = 'ld' + str(hyper_params['perc']) + '-no-teacher'
experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="akshaykvnit")
experiment.log_parameters(hyper_params)
optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params['learning_rate'])

os.makedirs('../saved_models/' + str(hyper_params['dataset']) + '/less_data' + str(hyper_params['perc']) + '/' + hyper_params['model'] + '_no_teacher', exist_ok = True)
savename = '../saved_models/' + str(hyper_params['dataset']) + '/less_data' + str(hyper_params['perc']) + '/' + hyper_params['model'] + '_no_teacher/model' + str(hyper_params['repeated']) + '.pt'
total_step = len(data.train_ds) // hyper_params['batch_size']
train_loss_list = list()
val_loss_list = list()
val_acc_list = list()
min_val = 0
for epoch in range(hyper_params['num_epochs']):
    trn = []
    net.train()
    for i, (images, labels) in enumerate(data.train_dl) :
        if torch.cuda.is_available():
            images = torch.autograd.Variable(images).cuda().float()
            labels = torch.autograd.Variable(labels).cuda()
        else : 
            images = torch.autograd.Variable(images).float()
            labels = torch.autograd.Variable(labels)

        y_pred = net(images)
        
        loss = F.cross_entropy(y_pred, labels)
        trn.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if i % 20 == 19 :
        #     print('epoch = ', epoch + 1, ' step = ', i + 1, ' of total steps ', total_step, ' loss = ', round(loss.item(), 4))
            
    train_loss = (sum(trn) / len(trn))
    train_loss_list.append(train_loss)
    
    net.eval()
    val = []
    with torch.no_grad() :
        for i, (images, labels) in enumerate(data.valid_dl) :
            if torch.cuda.is_available():
                images = torch.autograd.Variable(images).cuda().float()
                labels = torch.autograd.Variable(labels).cuda()
            else : 
                images = torch.autograd.Variable(images).float()
                labels = torch.autograd.Variable(labels)

            # Forward pass
            outputs = net(images)
            loss = F.cross_entropy(outputs, labels)
            val.append(loss)

    val_loss = (sum(val) / len(val)).item()
    val_loss_list.append(val_loss)
    val_acc = _get_accuracy(data.valid_dl, net)
    val_acc_list.append(val_acc)
    experiment.log_metric("train_loss", train_loss)
    experiment.log_metric("val_loss", val_loss)
    experiment.log_metric("val_acc", val_acc * 100)
    print('epoch : ', epoch + 1, ' / ', hyper_params['num_epochs'], ' | TL : ', round(train_loss, 4), ' | VL : ', round(val_loss, 4), ' | VA : ', round(val_acc * 100, 6))
    
    if (val_acc * 100) > min_val :
        print('saving model')
        min_val = val_acc * 100
        torch.save(net.state_dict(), savename)
        
# checking accuracy of best model
# net.load_state_dict(torch.load(savename))
# print('validation acc : ', _get_accuracy(data.valid_dl, net))
# print('test acc : ', _get_accuracy(data.test_dl, net))

# plt.plot(range(hyper_params['num_epochs']), train_loss_list, 'r', label = 'training_loss')
# plt.plot(range(hyper_params['num_epochs']), val_loss_list, 'b', label = 'validation_loss')
# plt.legend()
# plt.savefig('../figures/' + str(hyper_params['dataset']) + '/less_data_medium_no_teacher/training_losses' + str(hyper_params['repeated']) + '.jpg')
# plt.close()

# plt.plot(range(hyper_params['num_epochs']), val_acc_list, 'r', label = 'validation_accuracy')
# plt.legend()
# plt.savefig('../figures/' + str(hyper_params['dataset']) + '/less_data_medium_no_teacher/validation_acc' + str(hyper_params['repeated']) + '.jpg')
