from comet_ml import Experiment
from fastai.vision import *
import torch
from torchsummary import summary
from utils import _get_accuracy
from core import Flatten, conv_, conv_and_res, SaveFeatures
torch.cuda.set_device(0)

val = 'val'
sz = 224
stats = imagenet_stats
num_epochs = 100
batch_size = 64
dataset = 'imagenette'
path = untar_data(URLs.IMAGENETTE)

for repeated in range(1, 2) : 
    for stage in range(0, 4) :
        torch.manual_seed(repeated)
        torch.cuda.manual_seed(repeated)

        val = 'val'
        sz = 224
        stats = imagenet_stats

        # stage should be in 0 to 5 (5 for classifier stage)
        hyper_params = {
            "dataset": dataset,
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
        # no need to load model for 0th stage training
        if hyper_params['stage'] == 0 : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/small_stage' + str(hyper_params['stage']) + '/model' + str(hyper_params['repeated']) + '.pt'
        # separate if conditions for stage 1 and others because of irregular naming convention
        # in the student model.
        elif hyper_params['stage'] == 1 : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/small_stage0/model' + str(hyper_params['repeated']) + '.pt'
            net.load_state_dict(torch.load(filename, map_location = 'cpu'))
        else : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/small_stage' + str(hyper_params['stage']) + '/model' + str(hyper_params['repeated']) + '.pt'
            net.load_state_dict(torch.load(filename, map_location = 'cpu'))
        
        if torch.cuda.is_available() : 
            net = net.cuda()

        for name, param in net.named_parameters() : 
            param.requires_grad = False
            if name[0] == str(hyper_params['stage'] + 1) and hyper_params['stage'] != 0 :
                param.requires_grad = True
            elif name[0] == str(hyper_params['stage']) and hyper_params['stage'] == 0 : 
                param.requires_grad = True

        # saving outputs of all Basic Blocks
        mdl = learn.model
        sf = [SaveFeatures(m) for m in [mdl[0][2], mdl[0][4], mdl[0][5], mdl[0][6]]]
        sf2 = [SaveFeatures(m) for m in [net[0], net[2], net[3], net[4]]]
        
        experiment = Experiment(api_key="IOZ5docSriEdGRdQmdXQn9kpu", project_name="kd5", workspace="akshaykvnit")
        experiment.log_parameters(hyper_params)
        if hyper_params['stage'] == 0 : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/small_stage' + str(hyper_params['stage']) + '/model' + str(hyper_params['repeated']) + '.pt'
        else : 
            filename = '../saved_models/' + str(hyper_params['dataset']) + '/small_stage' + str(hyper_params['stage'] + 1) + '/model' + str(hyper_params['repeated']) + '.pt'
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

    # stage should be in 0 to 4 (4 for classifier stage)
    hyper_params = {
        "dataset": dataset,
        "stage": 4,
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
    filename = '../saved_models/' + str(hyper_params['dataset']) + '/small_stage4/model' + str(repeated) + '.pt'
    net.load_state_dict(torch.load(filename, map_location = 'cpu'))

    if torch.cuda.is_available() : 
        net = net.cuda()

    for name, param in net.named_parameters() : 
        param.requires_grad = False
        if name[0] == '7' or name[0] == '8':
            param.requires_grad = True
        
    experiment = Experiment(api_key="IOZ5docSriEdGRdQmdXQn9kpu", project_name="kd5", workspace="akshaykvnit")
    experiment.log_parameters(hyper_params)
    optimizer = torch.optim.Adam(net.parameters(), lr = hyper_params["learning_rate"])
    total_step = len(data.train_ds) // hyper_params["batch_size"]
    train_loss_list = list()
    val_loss_list = list()
    min_val = 0
    savename = '../saved_models/' + str(hyper_params['dataset']) + '/small_classifier/model' + str(repeated) + '.pt'
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
