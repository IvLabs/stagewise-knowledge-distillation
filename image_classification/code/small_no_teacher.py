from fastai.vision import *
import torch
from torchsummary import summary
from utils import _get_accuracy
torch.cuda.set_device(1)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

path = untar_data(URLs.IMAGEWOOF)

batch_size = 64
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, train = 'train', valid = 'val', bs = batch_size, size = 224, ds_tfms = tfms).normalize(imagenet_stats)

class Flatten(nn.Module) :
    def forward(self, input):
        return input.view(input.size(0), -1)

def conv2(ni, nf) : 
    return conv_layer(ni, nf, stride = 2)

class ResBlock(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.conv1 = conv_layer(nf,nf)
        
    def forward(self, x): 
        return (x + self.conv1(x))

def conv_and_res(ni, nf): 
    return nn.Sequential(conv2(ni, nf), ResBlock(nf))

def conv_(nf) : 
    return nn.Sequential(conv_layer(nf, nf), ResBlock(nf))
    
net = nn.Sequential(
    conv_layer(3, 64, ks = 7, stride = 2, padding = 3),
    nn.MaxPool2d(3, 2, padding = 1),
    conv_(64),
    conv_and_res(64, 128),
    conv_and_res(128, 256),
    AdaptiveConcatPool2d(),
    Flatten(),
    nn.Linear(2 * 256, 128),
    nn.Linear(128, 10)
)

if torch.cuda.is_available() : 
    net = net.cuda()
    print('Model on GPU')

optimizer = torch.optim.Adam(net.parameters(), lr = 1e-4)

num_epochs = 100
total_step = len(data.train_ds) // batch_size
train_loss_list = list()
val_loss_list = list()
val_acc_list = list()
min_val = 0
for epoch in range(num_epochs):
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
    print('epoch : ', epoch + 1, ' / ', num_epochs, ' | TL : ', round(train_loss, 4), ' | VL : ', round(val_loss, 4), ' | VA : ', round(val_acc * 100, 6))
    
    if (val_acc * 100) > min_val :
        print('saving model')
        min_val = val_acc * 100
        torch.save(net.state_dict(), '../saved_models/imagewoof/small_no_teacher/model0.pt')
        
# checking accuracy of best model
net.load_state_dict(torch.load('../saved_models/imagewoof/small_no_teacher/model0.pt'))
_get_accuracy(data.valid_dl, net)