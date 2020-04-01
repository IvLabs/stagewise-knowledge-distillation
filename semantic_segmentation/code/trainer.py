import torch
import torch.nn as nn
from comet_ml import Experiment
from helper import *
import torch.nn.functional as F
from tqdm import tqdm
from helper import get_savename


def train(model, train_loader, val_loader, num_classes, epoch, num_epochs, loss_function, optimiser, scheduler,
          savename, highest_iou):
    model.train()
    losses = list()
    val_losses = list()
    gpu1 = 'cuda:0'
    ious = list()
    pixel_accs = list()
    dices = list()
    max_iou = highest_iou
    savename2 = savename[: -3] + '_opt.pt'
    loop = tqdm(train_loader)
    num_steps = len(loop)
    for data, target in loop:
        model.train()
        model = model.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        optimiser.zero_grad()
        prediction = model(data)
        loss = loss_function(prediction, target)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, axis=1).squeeze(1)

        losses.append(loss.item())

        loss.backward()
        optimiser.step()
        scheduler.step()

        loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        loop.set_postfix(loss=loss.item())

    model.eval()
    for data, target in val_loader:
        model = model.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        prediction = model(data)
        val_loss = loss_function(prediction, target)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, axis=1).squeeze(1)

        ious.append(iou(target, prediction, num_classes=num_classes))
        dices.append(dice_coeff(target, prediction, num_classes=num_classes))
        pixel_accs.append(pixelwise_acc(prediction, target))
        val_losses.append(val_loss.item())

    avg_iou = sum(ious) / len(ious)
    avg_dice_coeff = sum(dices) / len(dices)
    avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)

    if avg_iou > max_iou:
        max_iou = avg_iou
        torch.save(model.state_dict(), savename)
        torch.save(optimiser.state_dict(), savename2)
        print('new max_iou', max_iou)

    print('avg_iou: ', avg_iou)
    print('avg_pixel_acc: ', avg_pixel_acc)
    print('avg_dice_coeff: ', avg_dice_coeff)

    avg_loss = sum(losses) / len(losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    return model, max_iou, avg_loss, avg_val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff


def unfreeze_trad(model, stage):
    # First asymmetrical stage
    if stage == 0:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith('encoder.conv') or name.startswith('encoder.bn') or name.startswith(
                    'encoder.layer0') or name.startswith('encoder.layer1') or name.startswith('encoder.layer2'):
                param.requires_grad = True
    elif stage == 1:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith('encoder.layer3') or name.startswith('encoder.layer4') or name.startswith(
                    'decoder.blocks.0') or name.startswith('decoder.blocks.1') or name.startswith('decoder.blocks.2'):
                param.requires_grad = True
    # Classifier stage
    elif stage == 2:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith('segmentation') or name.startswith('decoder.blocks.3') or name.startswith(
                    'decoder.blocks.4'):
                param.requires_grad = True
    # Freeze everything
    else:
        for name, param in model.named_parameters():
            param.requires_grad = False

    return model


def unfreeze(model, stage):
    # First asymmetrical stage
    if stage == 0:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith('encoder.conv') or name.startswith('encoder.bn'):
                param.requires_grad = True
    # Encoder stages
    elif stage > 0 and stage < 5:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith('encoder.layer' + str(stage)):
                param.requires_grad = True
    # Decoder stages
    elif stage > 4 and stage < 10:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith('decoder.blocks.' + str(stage - 5)):
                param.requires_grad = True
    # Classifier stage
    elif stage == 10:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if name.startswith('segmentation'):
                param.requires_grad = True
    # Freeze everything
    else:
        for name, param in model.named_parameters():
            param.requires_grad = False

    return model


def train_stage(model, teacher, stage, sf_student, sf_teacher, train_loader, val_loader, epoch, num_epochs,
                loss_function, optimiser, scheduler, savename, lowest_val):
    model.train()
    losses = list()
    val_losses = list()
    gpu1 = 'cuda:0'
    ious = list()
    pixel_accs = list()
    dices = list()
    lowest_val_loss = lowest_val
    savename2 = savename[: -3] + '_opt.pt'
    loop = tqdm(train_loader)
    num_steps = len(loop)
    for data, target in loop:
        model.train()
        model = model.to(gpu1)
        teacher = teacher.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        optimiser.zero_grad()
        _ = model(data)
        _ = teacher(data)

        loss = loss_function(sf_student[stage].features, sf_teacher[stage].features)
        #         prediction = F.softmax(prediction, dim = 1)
        #         prediction = torch.argmax(prediction, axis = 1).squeeze(1)

        losses.append(loss.item())

        loss.backward()
        optimiser.step()
        scheduler.step()

        loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        loop.set_postfix(loss=loss.item())

    model.eval()
    for data, target in val_loader:
        model = model.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        prediction = model(data)
        _ = teacher(data)
        val_loss = loss_function(sf_student[stage].features, sf_teacher[stage].features)

        val_losses.append(val_loss.item())

    avg_loss = sum(losses) / len(losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    if avg_val_loss < lowest_val_loss:
        lowest_val_loss = avg_val_loss
        torch.save(model.state_dict(), savename)
        torch.save(optimiser.state_dict(), savename2)
        print('new lowest_val_loss', lowest_val_loss)

    return model, lowest_val_loss, avg_loss, avg_val_loss


def train_trad_kd(model, teacher, sf_teacher, sf_student, train_loader, val_loader, num_classes, epoch, num_epochs,
                  loss_function1, loss_function2, optimiser, scheduler, savename, highest_iou):
    model.train()
    losses = list()
    val_losses = list()
    gpu1 = 'cuda:0'
    ious = list()
    pixel_accs = list()
    dices = list()
    max_iou = highest_iou
    savename2 = savename[: -3] + '_opt.pt'
    loop = tqdm(train_loader)
    num_steps = len(loop)
    for data, target in loop:
        model.train()
        model = model.to(gpu1)
        teacher = teacher.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        optimiser.zero_grad()
        prediction = model(data)
        _ = teacher(data)

        loss = loss_function1(prediction, target) + loss_function2(sf_teacher[0].features,
                                                                   sf_student[0].features) + loss_function2(
            sf_teacher[1].features, sf_student[1].features)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, axis=1).squeeze(1)

        losses.append(loss.item())

        loss.backward()
        optimiser.step()
        scheduler.step()

        loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        loop.set_postfix(loss=loss.item())

    model.eval()
    for data, target in val_loader:
        model = model.to(gpu1)
        teacher = teacher.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        prediction = model(data)
        _ = teacher(data)
        val_loss = loss_function1(prediction, target) + loss_function2(sf_teacher[0].features,
                                                                       sf_student[0].features) + loss_function2(
            sf_teacher[1].features, sf_student[1].features)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, axis=1).squeeze(1)

        ious.append(iou(target, prediction, num_classes=num_classes))
        dices.append(dice_coeff(target, prediction, num_classes=num_classes))
        pixel_accs.append(pixelwise_acc(prediction, target))
        val_losses.append(val_loss.item())

    avg_iou = sum(ious) / len(ious)
    avg_dice_coeff = sum(dices) / len(dices)
    avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)

    if avg_iou > max_iou:
        max_iou = avg_iou
        torch.save(model.state_dict(), savename)
        torch.save(optimiser.state_dict(), savename2)
        print('new max_iou', max_iou)

    print('avg_iou: ', avg_iou)
    print('avg_pixel_acc: ', avg_pixel_acc)
    print('avg_dice_coeff: ', avg_dice_coeff)

    avg_loss = sum(losses) / len(losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    return model, max_iou, avg_loss, avg_val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff


def train_simult(model, teacher, sf_teacher, sf_student, train_loader, val_loader, num_classes, epoch, num_epochs,
                 loss_function, loss_function2, optimiser, scheduler, savename, highest_iou):
    model.train()
    losses = list()
    val_losses = list()
    gpu1 = 'cuda:0'
    ious = list()
    pixel_accs = list()
    dices = list()
    max_iou = highest_iou
    savename2 = savename[: -3] + '_opt.pt'
    loop = tqdm(train_loader)
    num_steps = len(loop)
    for data, target in loop:
        model.train()
        model = model.to(gpu1)
        teacher = teacher.to(gpu1)

        data, target = data.float().to(gpu1), target.long().to(gpu1)

        optimiser.zero_grad()
        prediction = model(data)
        _ = teacher(data)
        loss = 0
        for i in range(10):
            loss += loss_function2(sf_teacher[i].features, sf_student[i].features)
        loss += loss_function(prediction, target)
        loss /= 11
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, axis=1).squeeze(1)

        losses.append(loss.item())

        loss.backward()
        optimiser.step()
        scheduler.step()

        loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        loop.set_postfix(loss=loss.item())

    model.eval()
    for data, target in val_loader:
        model = model.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        prediction = model(data)
        val_loss = loss_function(prediction, target)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, axis=1).squeeze(1)

        ious.append(iou(target, prediction, num_classes=num_classes))
        dices.append(dice_coeff(target, prediction, num_classes=num_classes))
        pixel_accs.append(pixelwise_acc(prediction, target))
        val_losses.append(val_loss.item())

    avg_iou = sum(ious) / len(ious)
    avg_dice_coeff = sum(dices) / len(dices)
    avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)

    if avg_iou > max_iou:
        max_iou = avg_iou
        torch.save(model.state_dict(), savename)
        torch.save(optimiser.state_dict(), savename2)
        print('new max_iou', max_iou)

    print('avg_iou: ', avg_iou)
    print('avg_pixel_acc: ', avg_pixel_acc)
    print('avg_dice_coeff: ', avg_dice_coeff)

    avg_loss = sum(losses) / len(losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    return model, max_iou, avg_loss, avg_val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff


def train_stagewise(hyper_params, teacher, student, sf_teacher, sf_student, trainloader, valloader, p=None):
    for stage in range(10):
        # update hyperparams dictionary
        hyper_params['stage'] = stage

        # Load previous stage model (except zeroth stage)
        if stage != 0:
            student.load_state_dict(torch.load(get_savename(hyper_params, mode='stagewise', p=p)))

        # Freeze all stages except current stage
        student = unfreeze(student, hyper_params['stage'])

        project_name = 'stagewise-' + hyper_params['model']
        experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="semseg_kd")
        experiment.log_parameters(hyper_params)

        optimizer = torch.optim.Adam(student.parameters(), lr=hyper_params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                        epochs=hyper_params['num_epochs'])
        criterion = nn.MSELoss()

        savename = '../saved_models/' + hyper_params['model'] + '/stage' + str(hyper_params['stage']) + '/model' + str(
            hyper_params['seed']) + '.pt'
        lowest_val_loss = 100
        losses = []
        for epoch in range(hyper_params['num_epochs']):
            student, lowest_val_loss, train_loss, val_loss = train_stage(model=student,
                                                                         teacher=teacher,
                                                                         stage=hyper_params['stage'],
                                                                         sf_student=sf_student,
                                                                         sf_teacher=sf_teacher,
                                                                         train_loader=trainloader,
                                                                         val_loader=valloader,
                                                                         loss_function=criterion,
                                                                         optimiser=optimizer,
                                                                         scheduler=scheduler,
                                                                         epoch=epoch,
                                                                         num_epochs=hyper_params['num_epochs'],
                                                                         savename=savename,
                                                                         lowest_val=lowest_val_loss
                                                                         )
            experiment.log_metric('train_loss', train_loss)
            experiment.log_metric('val_loss', val_loss)
        return teacher, student, hyper_params


def train_classifer(hyper_params, student, trainloader, valloader, p=None):
    # Freeze all stages except current stage
    student = unfreeze(student, hyper_params['stage'])
    project_name = 'stagewise-' + hyper_params['model']
    experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="semseg_kd")
    experiment.log_parameters(hyper_params)

    optimizer = torch.optim.Adam(student.parameters(), lr=hyper_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                    epochs=hyper_params['num_epochs'])
    criterion = nn.CrossEntropyLoss(ignore_index=11)

    savename = get_savename(hyper_params, mode='classifier', p=p)
    highest_iou = 0
    losses = []
    for epoch in range(hyper_params['num_epochs']):
        student, highest_iou, train_loss, val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff = train(model=student,
                                                                                                   train_loader=trainloader,
                                                                                                   val_loader=valloader,
                                                                                                   loss_function=criterion,
                                                                                                   optimiser=optimizer,
                                                                                                   scheduler=scheduler,
                                                                                                   epoch=epoch,
                                                                                                   num_epochs=
                                                                                                   hyper_params[
                                                                                                       'num_epochs'],
                                                                                                   savename=savename,
                                                                                                   highest_iou=highest_iou
                                                                                                   )
        experiment.log_metric('train_loss', train_loss)
        experiment.log_metric('val_loss', val_loss)
        experiment.log_metric('avg_iou', avg_iou)
        experiment.log_metric('avg_pixel_acc', avg_pixel_acc)
        experiment.log_metric('avg_dice_coeff', avg_dice_coeff)


def pretrain(hyper_params, unet, trainloader, valloader, p=None, dataset='camvid'):
    if p is not None:
        s = 'small' + str(p) +'-'
        less = '/less_data/'
        p = str(p) + '_'
    else:
        s = ''
        less = ''
        p = ''

    project_name = 'pretrain-'+ s + hyper_params['dataset'] + '-' + hyper_params['model']
    experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="semseg_kd")
    experiment.log_parameters(hyper_params)

    optimizer = torch.optim.Adam(unet.parameters(), lr=hyper_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                    epochs=hyper_params['num_epochs'])
    if dataset == 'camvid':
        criterion = nn.CrossEntropyLoss(ignore_index=11)
        num_classes = 12
    elif dataset == 'cityscapes':
        criterion = nn.CrossEntropyLoss(ignore_index=250)
        num_classes = 19

    savename = '../saved_models/' + hyper_params['dataset'] + less + hyper_params['model'] + '/pretrained_' + p + str(hyper_params['seed']) + '.pt'
    highest_iou = 0
    losses = []
    for epoch in range(hyper_params['num_epochs']):
        unet, highest_iou, train_loss, val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff = train(model=unet,
                                                                                                train_loader=trainloader,
                                                                                                val_loader=valloader,
                                                                                                num_classes=num_classes,
                                                                                                loss_function=criterion,
                                                                                                optimiser=optimizer,
                                                                                                scheduler=scheduler,
                                                                                                epoch=epoch,
                                                                                                num_epochs=hyper_params[
                                                                                                    'num_epochs'],
                                                                                                savename=savename,
                                                                                                highest_iou=highest_iou
                                                                                                )
        experiment.log_metric('train_loss', train_loss)
        experiment.log_metric('val_loss', val_loss)
        experiment.log_metric('avg_iou', avg_iou)
        experiment.log_metric('avg_pixel_acc', avg_pixel_acc)
        experiment.log_metric('avg_dice_coeff', avg_dice_coeff)

