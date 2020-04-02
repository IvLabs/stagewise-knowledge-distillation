import torch
import torch.nn as nn
import torch.nn.functional as F
from comet_ml import Experiment
from tqdm import tqdm

import models
from metrics import pixelwise_acc, dice_coeff, iou, mean_iou
from utils import get_savename


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


def train_stagewise(hyper_params, teacher, student, sf_teacher, sf_student, trainloader, valloader, args):
    for stage in range(10):
        # update hyperparams dictionary
        hyper_params['stage'] = stage

        # Load previous stage model (except zeroth stage)
        if stage != 0:
            student.load_state_dict(torch.load(get_savename(hyper_params, mode='stagewise', p=args.p)))

        # Freeze all stages except current stage
        student = unfreeze(student, hyper_params['stage'])

        project_name = 'stagewise-' + hyper_params['model']
        experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="semseg_kd")
        experiment.log_parameters(hyper_params)

        optimizer = torch.optim.Adam(student.parameters(), lr=hyper_params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                        epochs=hyper_params['num_epochs'])
        criterion = nn.MSELoss()

        savename = get_savename(hyper_params, dataset=args.d, mode='stagewise', p=args.p)
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

        hyper_params['stage'] = 10
        student.load_state_dict(torch.load(get_savename(hyper_params, mode='stagewise', p=args.p)))
        # Freeze all stages except current stage
        student = unfreeze(student, hyper_params['stage'])
        project_name = 'stagewise-' + hyper_params['model']
        experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name,
                                workspace="semseg_kd")
        experiment.log_parameters(hyper_params)

        optimizer = torch.optim.Adam(student.parameters(), lr=hyper_params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                        epochs=hyper_params['num_epochs'])
        criterion = nn.CrossEntropyLoss(ignore_index=11)

        savename = get_savename(hyper_params, dataset=args.d, mode='classifier', p=args.p)
        highest_iou = 0
        losses = []
        for epoch in range(hyper_params['num_epochs']):
            student, highest_iou, train_loss, val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff = train(
                model=student,
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


def pretrain(hyper_params, unet, trainloader, valloader, args):
    if args.p is not None:
        s = 'small' + str(args.p) + '-'
    else:
        s = ''

    project_name = 'pretrain-' + s + hyper_params['dataset'] + '-' + hyper_params['model']
    experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="semseg_kd")
    experiment.log_parameters(hyper_params)

    optimizer = torch.optim.Adam(unet.parameters(), lr=hyper_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                    epochs=hyper_params['num_epochs'])
    if args.d == 'camvid':
        criterion = nn.CrossEntropyLoss(ignore_index=11)
        num_classes = 12
    elif args.d == 'cityscapes':
        criterion = nn.CrossEntropyLoss(ignore_index=250)
        num_classes = 19

    savename = get_savename(hyper_params, dataset=args.d, mode='pretrain', p=args.p)
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


def train_simulataneous(hyper_params, teacher, student, sf_teacher, sf_student, trainloader, valloader, args):
    project_name = 'simultaneous-' + hyper_params['model']
    experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="semseg_kd")
    experiment.log_parameters(hyper_params)

    optimizer = torch.optim.Adam(student.parameters(), lr=hyper_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                    epochs=hyper_params['num_epochs'])
    criterion = nn.CrossEntropyLoss(ignore_index=11)
    criterion2 = nn.MSELoss()

    savename = get_savename(hyper_params, dataset=args.d, mode='simultaneous', p=args.p)
    highest_iou = 0
    for epoch in range(hyper_params['num_epochs']):
        _, _, train_loss, val_loss, avg_iou, avg_px_acc, avg_dice_coeff = train_simult(model=student,
                                                                                       teacher=teacher,
                                                                                       sf_teacher=sf_teacher,
                                                                                       sf_student=sf_student,
                                                                                       train_loader=trainloader,
                                                                                       val_loader=valloader,
                                                                                       num_classes=
                                                                                       hyper_params[
                                                                                           'num_classes'],
                                                                                       loss_function=criterion,
                                                                                       loss_function2=criterion2,
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
        experiment.log_metric('avg_pixel_acc', avg_px_acc)
        experiment.log_metric('avg_dice_coeff', avg_dice_coeff)


def train_traditional(hyper_params, teacher, student, sf_teacher, sf_student, trainloader, valloader, args):
    for stage in range(2):
        # update hyperparams dictionary
        hyper_params['stage'] = stage

        # Load previous stage model (except zeroth stage)
        if stage != 0:
            savename = '../saved_models/camvid/trad_kd_new/' + hyper_params['model'] + '/stage' + str(
                hyper_params['stage'] - 1) + '/model' + str(hyper_params['seed']) + '.pt'
            student.load_state_dict(torch.load(savename))

        # Freeze all stages except current stage
        student = unfreeze_trad(student, hyper_params['stage'])

        project_name = 'new-trad-kd-' + hyper_params['model']
        experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="semseg_kd")
        experiment.log_parameters(hyper_params)

        optimizer = torch.optim.Adam(student.parameters(), lr=hyper_params['learning_rate'])
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                        epochs=hyper_params['num_epochs'])
        criterion = nn.MSELoss()

        savename = get_savename(hyper_params, args.d, mode='traditional-kd', p=args.p)
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

    # Classifier training
    hyper_params['stage'] = 2
    savename = get_savename(hyper_params, args.d, mode='traditional-kd', p=args.p)
    student.load_state_dict(torch.load(savename))

    # Freeze all stages except current stage
    student = unfreeze_trad(student, hyper_params['stage'])

    project_name = 'new-trad-kd-' + hyper_params['model']
    experiment = Experiment(api_key="1jNZ1sunRoAoI2TyremCNnYLO", project_name=project_name, workspace="semseg_kd")
    experiment.log_parameters(hyper_params)

    optimizer = torch.optim.Adam(student.parameters(), lr=hyper_params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, steps_per_epoch=len(trainloader),
                                                    epochs=hyper_params['num_epochs'])
    criterion = nn.CrossEntropyLoss(ignore_index=11)

    savename = get_savename(hyper_params, args.d, mode='traditional-kd', p=args.p)
    highest_iou = 0
    losses = []
    for epoch in range(hyper_params['num_epochs']):
        student, highest_iou, train_loss, val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff = train(model=student,
                                                                                                   train_loader=trainloader,
                                                                                                   val_loader=valloader,
                                                                                                   num_classes=12,
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


def evaluate(valloader, args, params, mode):
    print(f'Fractional data results for {mode}')
    for perc in [10, 20, 30, 40]:
        print('perc : ', perc)
        for model_name in ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26']:
            params['model'] = model_name
            print('model : ', model_name)
            unet = models.unet.Unet(model_name, classes=params['num_classes'], encoder_weights=None).cuda()
            unet.load_state_dict(
                torch.load(get_savename(params, dataset=args.d, mode=mode, p=perc)))
            current_val_iou = mean_iou(unet, valloader)
            print(round(current_val_iou, 5))

    print(f'Full data results for {mode}')
    for model_name in ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26']:
        print('model : ', model_name)
        unet = models.unet.Unet(model_name, classes=params['num_classes'], encoder_weights=None).cuda()
        unet.load_state_dict(torch.load(get_savename(params, dataset=args.d, mode=mode)))
        current_val_iou = mean_iou(unet, valloader)
        print(round(current_val_iou, 5))
