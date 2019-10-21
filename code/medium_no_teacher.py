import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from fastai.vision import *
import torch
from torchsummary import summary
from utils import _get_accuracy
from models.custom_resnet import *
torch.cuda.set_device(1)
path = untar_data(URLs.IMAGENETTE)
batch_size = 64
num_epochs = 100
dataset = 'imagenette'
model_name = 'resnet18'

for repeated in range(0, 1) : 
    torch.manual_seed(repeated)
    torch.cuda.manual_seed(repeated)

    val = 'val'
    sz = 224
    stats = imagenet_stats

    # stage should be in 0 to 5 (5 for classifier stage)
    hyper_params = {
        "model": model_name,
        "dataset": dataset,
        "repeated": repeated,
        "num_classes": 10,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": 1e-4
        }

    tfms = get_transforms(do_flip=False)
    load_name = str(hyper_params['dataset'])
    if hyper_params['dataset'] == 'cifar10' : 
        val = 'test'
        sz = 32
        stats = cifar_stats
        load_name = str(hyper_params['dataset'])[ : -2]

    data = ImageDataBunch.from_folder(path, train = 'train', valid = val, bs = hyper_params["batch_size"], size = sz, ds_tfms = tfms).normalize(stats)

#     net = _resnet('resnet14', BasicBlock, [2, 2, 1, 1], pretrained = False, progress = False)
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
        print('Model on GPU')

    optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params['learning_rate'])

    savename = '../saved_models/'+ str(hyper_params['dataset']) + '/' + str(hyper_params['model']) + '_no_teacher/model' + str(repeated) + '.pt'
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

    #         if i % 50 == 49 :
    #             print('epoch = ', epoch + 1, ' step = ', i + 1, ' of total steps ', total_step, ' loss = ', round(loss.item(), 4))
                
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
        print('epoch : ', epoch + 1, ' / ', num_epochs, ' | TL : ', round(train_loss, 4), ' | VL : ', round(val_loss, 4), ' | VA : ', round(val_acc * 100, 6))
        
        if (val_acc * 100) > min_val :
            print('saving model')
            min_val = val_acc * 100
            torch.save(net.state_dict(), savename)

#     plt.plot(range(hyper_params['num_epochs']), train_loss_list, 'r', label = 'training_loss')
#     plt.plot(range(hyper_params['num_epochs']), val_loss_list, 'b', label = 'validation_loss')
#     plt.legend()
#     plt.savefig('../figures/' + str(hyper_params['dataset']) + '/resnet14_no_teacher/training_losses' + str(repeated) + '.jpg')
#     plt.close()

#     plt.plot(range(hyper_params['num_epochs']), val_acc_list, 'r', label = 'validation_accuracy')
#     plt.legend()
#     plt.savefig('../figures/' + str(hyper_params['dataset']) + '/resnet14_no_teacher/validation_acc' + str(repeated) + '.jpg')
        
# checking accuracy of best model
net.load_state_dict(torch.load('../saved_models/' + str(hyper_params['dataset']) + '/' + str(hyper_params['model']) + '_no_teacher/model0.pt'))
print(_get_accuracy(data.valid_dl, net))
