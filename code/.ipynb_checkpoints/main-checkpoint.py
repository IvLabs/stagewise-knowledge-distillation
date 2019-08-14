from fastai.vision import *
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
torch.cuda.set_device(0)

path = untar_data(URLs.IMAGENETTE)

batch_size = 64
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, train = 'train', valid = 'val', bs = batch_size, size = 224, ds_tfms = tfms).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet34, metrics = accuracy)
learn = learn.load('unfreeze_imagenet_bs64')
# learn.summary()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

net = nn.Sequential(
    nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size = 2, stride = 2),
    Flatten(),
    nn.Linear(512 * 7 * 7, 512),
    nn.Linear(512, 10)
)

if torch.cuda.is_available() : 
    net = net.cuda()
    print('Model on GPU')
    
class SaveFeatures :
    def __init__(self, m) : 
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp) : 
        self.features = outp
    def remove(self) :
        self.handle.remove()
        
# saving outputs of all Basic Blocks
mdl = learn.model
sf = [SaveFeatures(m) for m in [mdl[0][2], mdl[0][4]]]
sf2 = [SaveFeatures(m) for m in [net[1], net[3]]]

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
        
        for k in range(3) : 
            loss += F.mse_loss(sf[k].features, sf2[k].features)
        
        loss += F.cross_entropy(y_pred, labels)
        trn.append(loss.item() / 4)

        optimizer.zero_grad()
        loss.backward()
#         torch.nn.utils.clip_grad_value_(net.parameters(), 10)
        optimizer.step()

        if i % 50 == 49 :
            print('epoch = ', epoch, ' step = ', i + 1, ' of total steps ', total_step, ' loss = ', loss.item() / 4)
            
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

    print('epoch : ', epoch + 1, ' / ', num_epochs, ' | TL : ', train_loss, ' | VL : ', val_loss, ' | VA : ', val_acc * 100)
    val_acc_list.append(val_acc)
    if (val_acc * 100) > min_val :
        print('saving model')
        min_val = val_acc * 100
        torch.save(net.state_dict(), '../saved_models/model5.pt')
        
# checking accuracy of best model
net.load_state_dict(torch.load('../saved_models/model5.pt'))
_get_accuracy(data.valid_dl, net)

plt.plot(range(100), train_loss_list, 'r', label = 'training_loss')
plt.plot(range(100), val_loss_list, 'b', label = 'validation_loss')
plt.legend()
plt.savefig('../figures/training_losses5.jpg')
plt.show()
plt.close()

plt.plot(range(100), val_acc_list, 'r', label = 'validation_accuracy')
plt.legend()
plt.savefig('../figures/validation_acc5.jpg')
plt.show()
