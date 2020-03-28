from fastai.vision import *
import torch
from torchsummary import summary
torch.cuda.set_device(0)

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

for repeated in range(3, 5) : 
    torch.manual_seed(repeated)
    torch.cuda.manual_seed(repeated)

    # stage should be in 0 to 5 (5 for classifier stage)
    hyper_params = {
        "stage": 5,
        "repeated": repeated,
        "num_classes": 10,
        "batch_size": 64,
        "num_epochs": 100,
        "learning_rate": 1e-4
    }

    path = untar_data(URLs.IMAGENETTE)
    tfms = get_transforms(do_flip=False)
    data = ImageDataBunch.from_folder(path, train = 'train', valid = 'val', bs = hyper_params["batch_size"], size = 224, ds_tfms = tfms).normalize(imagenet_stats)
    
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
        nn.Linear(128, hyper_params["num_classes"])
    )

    net.cpu()
    filename = '../saved_models/small_stage4/model' + str(repeated) + '.pt'
    net.load_state_dict(torch.load(filename, map_location = 'cpu'))

    if torch.cuda.is_available() : 
        net = net.cuda()
        print('Model on GPU')

    for name, param in net.named_parameters() : 
        param.requires_grad = False
        if name[0] == '7' or name[0] == '8':
            param.requires_grad = True
        
    optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params["learning_rate"])
    total_step = len(data.train_ds) // hyper_params["batch_size"]
    train_loss_list = list()
    val_loss_list = list()
    min_val = 0
    savename = '../saved_models/small_classifier/model' + str(repeated) + '.pt'
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
    #         torch.nn.utils.clip_grad_value_(net.parameters(), 10)
            optimizer.step()

            if i % 50 == 49 :
                print('epoch = ', epoch, ' step = ', i + 1, ' of total steps ', total_step, ' loss = ', loss.item())

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

        print('epoch : ', epoch + 1, ' / ', hyper_params["num_epochs"], ' | TL : ', train_loss, ' | VL : ', val_loss, ' | VA : ', val_acc * 100)

        if (val_acc * 100) > min_val :
            print('saving model')
            min_val = val_acc * 100
            torch.save(net.state_dict(), savename)
