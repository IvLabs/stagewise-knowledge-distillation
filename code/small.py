from fastai.vision import *
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
torch.cuda.set_device(0)
torch.manual_seed(5)
torch.cuda.manual_seed(5)

path = untar_data(URLs.IMAGENETTE)

batch_size = 64
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, train = 'train', valid = 'val', bs = batch_size, size = 224, ds_tfms = tfms).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet34, metrics = accuracy)
learn = learn.load('unfreeze_imagenet_bs64')
# learn.summary()

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
    
# net.load_state_dict(torch.load('../saved_models/medium_no_teacher/model.pt'))

class SaveFeatures :
    def __init__(self, m) : 
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp) : 
        self.features = outp
    def remove(self) :
        self.handle.remove()
        
# saving outputs of all Basic Blocks
mdl = learn.model
# for all 5 feature maps
sf = [SaveFeatures(m) for m in [mdl[0][2], mdl[0][4], mdl[0][5], mdl[0][6]]]
sf2 = [SaveFeatures(m) for m in [net[0], net[2], net[3], net[4]]]
num_fm = 3

def _get_accuracy(dataloader, Net):
    total = 0
    correct = 0
    Net.eval()
    for i, (images, labels) in enumerate(dataloader):
        images = torch.autograd.Variable(images).float()
        labels = torch.autograd.Variable(labels).float()
        
        if torch.cuda.is_available() : 
            images = images.cuda()
            labels = labels.cuda()

        outputs = Net.forward(images)
        outputs = F.log_softmax(outputs, dim = 1)

        _, pred_ind = torch.max(outputs, 1)
        
        # converting to numpy arrays
        labels = labels.data.cpu().numpy()
        pred_ind = pred_ind.data.cpu().numpy()
        
        # get difference
        diff_ind = labels - pred_ind
        # correctly classified will be 1 and will get added
        # incorrectly classified will be 0
        correct += np.count_nonzero(diff_ind == 0)
        total += len(diff_ind)

    accuracy = correct / total
    # print(len(diff_ind))
    return accuracy

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
        loss = 0.0
        if torch.cuda.is_available():
            images = torch.autograd.Variable(images).cuda().float()
            labels = torch.autograd.Variable(labels).cuda()
        else : 
            images = torch.autograd.Variable(images).float()
            labels = torch.autograd.Variable(labels)

        y_pred = net(images)
        y_pred2 = mdl(images)
        
        for k in range(num_fm) : 
            loss += F.mse_loss(sf[k].features, sf2[k].features)
        
        loss += F.cross_entropy(y_pred, labels)
        trn.append(loss.item() / (num_fm + 1))

        optimizer.zero_grad()
        loss.backward()
#         torch.nn.utils.clip_grad_value_(net.parameters(), 10)
        optimizer.step()

        if i % 50 == 49 :
            print('epoch = ', epoch + 1, ' step = ', i + 1, ' of total steps ', total_step, ' loss = ', round(loss.item() / (num_fm + 1), 4))
            
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

    print('epoch : ', epoch + 1, ' / ', num_epochs, ' | TL : ', round(train_loss, 4), ' | VL : ', round(val_loss, 4), ' | VA : ', round(val_acc * 100, 6))
    val_acc_list.append(val_acc)
    if (val_acc * 100) > min_val :
        print('saving model')
        min_val = val_acc * 100
        torch.save(net.state_dict(), '../saved_models/small_3fm/model4.pt')
        
# checking accuracy of best model
net.load_state_dict(torch.load('../saved_models/small_3fm/model4.pt'))
print(_get_accuracy(data.valid_dl, net))

plt.plot(range(num_epochs), train_loss_list, 'r', label = 'training_loss')
plt.plot(range(num_epochs), val_loss_list, 'b', label = 'validation_loss')
plt.legend()
plt.savefig('../figures/small_3fm/training_losses4.jpg')
plt.close()

plt.plot(range(num_epochs), val_acc_list, 'r', label = 'validation_accuracy')
plt.legend()
plt.savefig('../figures/small_3fm/validation_acc4.jpg')
