import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from comet_ml import Experiment
from fastai.vision import *
import torch
from torchsummary import summary
from core import SaveFeatures
from utils import _get_accuracy
from models.custom_resnet import *
torch.cuda.set_device(0)

val = 'val'
sz = 224
stats = imagenet_stats
num_epochs = 100
batch_size = 64
dataset = 'imagenette'
model_name = 'resnet26'
path = untar_data(URLs.IMAGENETTE)

for repeated in range(0, 1) : 
    for stage in range(5) :
        torch.manual_seed(repeated)
        torch.cuda.manual_seed(repeated)

        val = 'val'
        sz = 224
        stats = imagenet_stats

        # stage should be in 0 to 5 (5 for classifier stage)
        hyper_params = {
            "dataset": dataset,
            "model": model_name,
            "stage": stage,
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
        
        learn = cnn_learner(data, models.resnet34, metrics = accuracy)
        learn = learn.load('resnet34_' + load_name + '_bs64')
        learn.freeze()

        # net = _resnet('resnet14', BasicBlock, [2, 2, 1, 1], pretrained = False, progress = False)
        if model_name == 'resnet10' :
            net = resnet10(pretrained = False, progress = False)
        elif model_name == 'resnet14' : 
            net = resnet14(pretrained = False, progress = False)
        elif model_name == 'resnet20' :
            net = resnet20(pretrained = False, progress = False)
        elif model_name == 'resnet26' :
            net = resnet26(pretrained = False, progress = False)

        net.cpu()
        # no need to load model for 0th stage training
        if hyper_params['stage'] == 0 : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/' + str(hyper_params['model']) + '_stage' + str(hyper_params['stage']) + '/model' + str(hyper_params['repeated']) + '.pt'
        # separate if conditions for stage 1 and others because of irregular naming convention
        # in the student model.
        elif hyper_params['stage'] == 1 : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/' + str(hyper_params['model']) + '_stage0/model' + str(hyper_params['repeated']) + '.pt'
            net.load_state_dict(torch.load(filename, map_location = 'cpu'))
        else : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/' + str(hyper_params['model']) + '_stage' + str(hyper_params['stage']) + '/model' + str(hyper_params['repeated']) + '.pt'
            net.load_state_dict(torch.load(filename, map_location = 'cpu'))
        
        if torch.cuda.is_available() : 
            net = net.cuda()

        for name, param in net.named_parameters() : 
            param.requires_grad = False
            if name[5] == str(hyper_params['stage']) and hyper_params['stage'] != 0 :
                param.requires_grad = True
            elif (name[0] == 'b' or name[0] == 'c') and hyper_params['stage'] == 0 : 
                param.requires_grad = True

        # saving outputs of all Basic Blocks
        mdl = learn.model
        sf = [SaveFeatures(m) for m in [mdl[0][2], mdl[0][4], mdl[0][5], mdl[0][6], mdl[0][7]]]
        sf2 = [SaveFeatures(m) for m in [net.relu2, net.layer1, net.layer2, net.layer3, net.layer4]]
        
        experiment = Experiment(api_key="IOZ5docSriEdGRdQmdXQn9kpu", project_name="kd_resnet26_imagenette", workspace="akshaykvnit")
        experiment.log_parameters(hyper_params)
        if hyper_params['stage'] == 0 : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/' + str(hyper_params['model']) + '_stage' + str(hyper_params['stage']) + '/model' + str(hyper_params['repeated']) + '.pt'
        else : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/' + str(hyper_params['model']) + '_stage' + str(hyper_params['stage'] + 1) + '/model' + str(hyper_params['repeated']) + '.pt'
        optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params["learning_rate"])
        total_step = len(data.train_ds) // hyper_params["batch_size"]
        train_loss_list = list()
        val_loss_list = list()
        min_val = 100

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
                y_pred2 = mdl(images)

                loss = F.mse_loss(sf2[hyper_params["stage"]].features, sf[hyper_params["stage"]].features)
                trn.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # if i % 50 == 49 :
                    # print('epoch = ', epoch + 1, ' step = ', i + 1, ' of total steps ', total_step, ' loss = ', round(loss.item(), 6))

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
                    y_pred2 = mdl(images)
                    loss = F.mse_loss(sf[hyper_params["stage"]].features, sf2[hyper_params["stage"]].features)
                    val.append(loss.item())

            val_loss = sum(val) / len(val)
            val_loss_list.append(val_loss)
            
            if (epoch + 1) % 5 == 0 : 
                print('repetition : ', hyper_params["repeated"], ' | stage : ', hyper_params["stage"])
                print('epoch : ', epoch + 1, ' / ', hyper_params["num_epochs"], ' | TL : ', round(train_loss, 6), ' | VL : ', round(val_loss, 6))
            
            experiment.log_metric("train_loss", train_loss)
            experiment.log_metric("val_loss", val_loss)
            
            if val_loss < min_val :
                # print('saving model')
                min_val = val_loss
                torch.save(net.state_dict(), filename)


    # Classifier training
    torch.manual_seed(repeated)
    torch.cuda.manual_seed(repeated)
    val = 'val'
    sz = 224
    stats = imagenet_stats

    # stage should be in 0 to 5 (5 for classifier stage)
    hyper_params = {
        "dataset": dataset,
        "model": model_name,
        "stage": 5,
        "repeated": repeated,
        "num_classes": 10,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": 1e-4
    }
    load_name = str(hyper_params['dataset'])
    tfms = get_transforms(do_flip=False)
    if hyper_params['dataset'] == 'cifar10' : 
        val = 'test'
        sz = 32
        stats = cifar_stats
        load_name = str(hyper_params['dataset'])[ : -2]

    data = ImageDataBunch.from_folder(path, train = 'train', valid = val, bs = hyper_params["batch_size"], size = sz, ds_tfms = tfms).normalize(stats)

    learn = cnn_learner(data, models.resnet34, metrics = accuracy)
    learn = learn.load('resnet34_' + load_name + '_bs64')
    learn.freeze()
    # learn.summary()

    # net = _resnet('resnet14', BasicBlock, [2, 2, 1, 1], pretrained = False, progress = False)
    if model_name == 'resnet10' :
        net = resnet10(pretrained = False, progress = False)
    elif model_name == 'resnet14' : 
        net = resnet14(pretrained = False, progress = False)
    elif model_name == 'resnet20' :
        net = resnet20(pretrained = False, progress = False)
    elif model_name == 'resnet26' :
        net = resnet26(pretrained = False, progress = False)
    
    net.cpu()
    filename = '../saved_models/' + str(hyper_params['dataset']) + '/' + str(hyper_params['model']) + '_stage5/model' + str(repeated) + '.pt'
    net.load_state_dict(torch.load(filename, map_location = 'cpu'))

    if torch.cuda.is_available() : 
        net = net.cuda()

    for name, param in net.named_parameters() : 
        param.requires_grad = False
        if name[0] == 'f' :
            param.requires_grad = True
        
    experiment = Experiment(api_key="IOZ5docSriEdGRdQmdXQn9kpu", project_name="kd_resnet26_imagenette", workspace="akshaykvnit")
    experiment.log_parameters(hyper_params)
    optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params["learning_rate"])
    total_step = len(data.train_ds) // hyper_params["batch_size"]
    train_loss_list = list()
    val_loss_list = list()
    min_val = 0
    savename = '../saved_models/' + str(hyper_params['dataset']) + '/' + hyper_params['model'] + '_classifier/model' + str(repeated) + '.pt'
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
        experiment.log_metric("train_loss", train_loss)
        experiment.log_metric("val_loss", val_loss)
        experiment.log_metric("val_acc", val_acc * 100)

        print('epoch : ', epoch + 1, ' / ', hyper_params["num_epochs"], ' | TL : ', round(train_loss, 6), ' | VL : ', round(val_loss, 6), ' | VA : ', round(val_acc * 100, 6))

        if (val_acc * 100) > min_val :
            print('saving model')
            min_val = val_acc * 100
            torch.save(net.state_dict(), savename)
