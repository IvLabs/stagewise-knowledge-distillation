from fastai.vision import *
import torch
from torchsummary import summary
from models.custom_resnet import _resnet, Bottleneck
torch.cuda.set_device(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)

path = untar_data(URLs.IMAGENETTE)

batch_size = 32
tfms = get_transforms(do_flip=False)
data = ImageDataBunch.from_folder(path, train = 'train', valid = 'val', bs = batch_size, size = 224, ds_tfms = tfms).normalize(imagenet_stats)

net = _resnet('resnet50', Bottleneck, [2, 2, 2, 1], pretrained = False, progress = False)
net.cpu()
net.load_state_dict(torch.load('../saved_models/large_no_teacher/model0.pt', map_location = 'cpu'))

if torch.cuda.is_available() : 
    net = net.cuda()
    print('Model on GPU')

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
#         torch.nn.utils.clip_grad_value_(net.parameters(), 10)
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
        torch.save(net.state_dict(), '../saved_models/large_no_teacher/model1.pt')
        
# checking accuracy of best model
net.load_state_dict(torch.load('../saved_models/large_no_teacher/model1.pt'))
_get_accuracy(data.valid_dl, net)

# plt.plot(range(100), train_loss_list, 'r', label = 'training_loss')
# plt.plot(range(100), val_loss_list, 'b', label = 'validation_loss')
# plt.legend()
# plt.savefig('../figures/small_no_teacher/training_losses4.jpg')
# plt.close()

# plt.plot(range(100), val_acc_list, 'r', label = 'validation_accuracy')
# plt.legend()
# plt.savefig('../figures/small_no_teacher/validation_acc4.jpg')
