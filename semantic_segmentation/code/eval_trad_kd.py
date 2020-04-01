import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import models
from dataset import CamVid
from helper import *
torch.cuda.set_device(0)

num_classes = 12
DATA_DIR = '../data/CamVid/'
train_dataset = CamVid(DATA_DIR, mode='train')
valid_dataset = CamVid(DATA_DIR, mode='val')
test_dataset = CamVid(DATA_DIR, mode='test')

trainloader = DataLoader(train_dataset, batch_size = 8, shuffle = True, drop_last = True)
valloader = DataLoader(valid_dataset, batch_size = 1, shuffle = False)
testloader = DataLoader(test_dataset, batch_size = 1, shuffle = False)

def mean_iou(model, dataloader) : 
    gpu1 = 'cuda:0'
    ious = list()
    for i, (data, target) in enumerate(dataloader) : 
        data, target = data.float().to(gpu1), target.long().to(gpu1)
        prediction = model(data)
        prediction = F.softmax(prediction, dim = 1)
        prediction = torch.argmax(prediction, axis = 1).squeeze(1)
        ious.append(iou(target, prediction, num_classes = 11))
        
    return (sum(ious) / len(ious))

for model_name in ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26'] : 
    train_ious = list()
    val_ious = list()
    test_ious = list()
    unet = models.unet.Unet(model_name, classes = 12, encoder_weights = None).cuda()
    unet.load_state_dict(torch.load('../saved_models/camvid/trad_kd_new/' + model_name + '/classifier/model0.pt'))
    current_val_iou = mean_iou(unet, valloader)
    current_train_iou = mean_iou(unet, trainloader)
    current_test_iou = mean_iou(unet, testloader)
    train_ious.append(current_train_iou)
    val_ious.append(current_val_iou)
    test_ious.append(current_test_iou)
    print(round(current_train_iou, 5), round(current_val_iou, 5), round(current_test_iou, 5))