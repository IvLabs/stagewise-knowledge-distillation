from fastai.vision import *
import torch
import warnings
from torchsummary import summary
# from models.custom_resnet import _resnet, Bottleneck
from models.resnet_cifar import *
from utils import _get_accuracy
torch.cuda.set_device(0)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# stage should be in 0 to 6 (5 for classifier stage, 6 for finetuning stage)
hyper_params = {
    "dataset": 'cifar10',
    "stage": 6,
    "repeated": 0,
    "num_classes": 10,
    "batch_size": 64,
    "num_epochs": 100,
    "learning_rate": 1e-8
}

path = untar_data(URLs.CIFAR)

tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, train = 'train', valid = 'test', bs = hyper_params["batch_size"], size = 32, ds_tfms = tfms).normalize(cifar_stats)

filename = '../saved_models/' + hyper_params['dataset'] + '/medium_classifier/model0.pt'
net = resnet14_cifar()
net.cpu()
net.load_state_dict(torch.load(filename, map_location = 'cpu'))
if torch.cuda.is_available() : 
    net.cuda()
    
optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params['learning_rate'])
total_step = len(data.train_ds) // hyper_params["batch_size"]
train_loss_list = list()
val_loss_list = list()
min_val = 0
savename = '../saved_models/' + str(hyper_params['dataset']) + '/resnet14_classifier/model2.pt'
for epoch in range(hyper_params["num_epochs"]):
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

        # if i % 50 == 49 :
            # print('epoch = ', epoch, ' step = ', i + 1, ' of total steps ', total_step, ' loss = ', loss.item())

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
            y_pred = net(images)

            loss = F.cross_entropy(y_pred, labels)
            val.append(loss.item())

    val_loss = sum(val) / len(val)
    val_loss_list.append(val_loss)
    val_acc = _get_accuracy(data.valid_dl, net)

    print('epoch : ', epoch + 1, ' / ', hyper_params["num_epochs"], ' | TL : ', round(train_loss, 6), ' | VL : ', round(val_loss, 6), ' | VA : ', round(val_acc * 100, 6))

    if (val_acc * 100) > min_val :
        print('saving model')
        min_val = val_acc * 100
        torch.save(net.state_dict(), savename)