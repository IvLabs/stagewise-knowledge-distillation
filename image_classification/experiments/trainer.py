from comet_ml import Experiment
from tqdm import tqdm

from fastai.vision import *

from image_classification.utils.utils import *

def train(student, teacher, data, sf_teacher, sf_student, loss_function, loss_function2, optimizer, hyper_params, epoch, savename, best_val_acc, expt=None):
    loop = tqdm(data.train_dl)
    max_val_acc = best_val_acc
    gpu = hyper_params['gpu']
    student.train()
    student = student.to(gpu)
    if teacher is not None:
        teacher.eval()
        teacher = teacher.to(gpu)
    trn = list()
    for images, labels in loop:
        if gpu != 'cpu':
            images = torch.autograd.Variable(images).to(gpu).float()
            labels = torch.autograd.Variable(labels).to(gpu)
        else:
            images = torch.autograd.Variable(images).float()
            labels = torch.autograd.Variable(labels)

        y_pred = student(images)
        if teacher is not None:
            _ = teacher(images)

        # classifier training
        if teacher is None:
            loss = loss_function(y_pred, labels)
        # stage training (and assuming sf_teacher and sf_student are given)
        elif loss_function2 is None:
            if expt == 'fsp-kd':
                loss = 0
                # 4 intermediate feature maps and taken 2 at a time (thus 3)
                for k in range(3):
                    loss += loss_function(fsp_matrix(sf_teacher[k].features, sf_teacher[k + 1].features),
                                          fsp_matrix(sf_student[k].features, sf_student[k + 1].features))
                loss /= 3
            else:
                loss = loss_function(sf_student[hyper_params['stage']].features, sf_teacher[hyper_params['stage']].features)
        # attention transfer KD
        elif expt == 'attention-kd':
            loss = loss_function(y_pred, labels)
            for k in range(4):
                loss += loss_function2(at(sf_student[k].features), at(sf_teacher[k].features))
            loss /= 5
        # 2 loss functions and student and teacher are given -> simultaneous training
        else:
            loss = loss_function(y_pred, labels)
            for k in range(5):
                loss += loss_function2(sf_student[k].features, sf_teacher[k].features)
            # normalizing factor (doesn't affect optimization theoretically)
            loss /= 6

        trn.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_description('Epoch {}/{}'.format(epoch + 1, hyper_params['num_epochs']))
        loop.set_postfix(loss=loss.item())

    train_loss = (sum(trn) / len(trn))

    student.eval()
    val = list()
    with torch.no_grad():
        correct = 0
        total = 0
        for _, (images, labels) in enumerate(data.valid_dl):
            if gpu != 'cpu':
                images = torch.autograd.Variable(images).to(gpu).float()
                labels = torch.autograd.Variable(labels).to(gpu)
            else:
                images = torch.autograd.Variable(images).float()
                labels = torch.autograd.Variable(labels)

            y_pred = student(images)
            if teacher is not None:
                _ = teacher(images)

            # classifier training
            if teacher is None:
                loss = loss_function(y_pred, labels)
                y_pred = F.log_softmax(y_pred, dim = 1)

                _, pred_ind = torch.max(y_pred, 1)

                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()
            # stage training
            elif loss_function2 is None:
                if expt == 'fsp-kd':
                    loss = 0
                    # 4 intermediate feature maps and taken 2 at a time (thus 3)
                    for k in range(3):
                        loss += loss_function(fsp_matrix(sf_teacher[k].features, sf_teacher[k + 1].features),
                                              fsp_matrix(sf_student[k].features, sf_student[k + 1].features))
                    loss /= 3
                else:
                    loss = loss_function(sf_student[hyper_params['stage']].features, sf_teacher[hyper_params['stage']].features)
            # simultaneous training or attention KD
            else:
                loss = loss_function(y_pred, labels)
                y_pred = F.log_softmax(y_pred, dim = 1)

                _, pred_ind = torch.max(y_pred, 1)

                total += labels.size(0)
                correct += (pred_ind == labels).sum().item()

            val.append(loss.item())

    val_loss = (sum(val) / len(val))
    if total > 0:
        val_acc = correct / total
    else:
        val_acc = None

    # classifier training
    if teacher is None:
        if (val_acc * 100) > max_val_acc :
            print(f'higher valid acc obtained: {val_acc * 100}')
            max_val_acc = val_acc * 100
            torch.save(student.state_dict(), savename)
    # stage training
    elif loss_function2 is None:
        if val_loss < max_val_acc :
            print(f'lower valid loss obtained: {val_loss}')
            max_val_acc = val_loss
            torch.save(student.state_dict(), savename)
    # simultaneous training or attention kd
    else:
        if (val_acc * 100) > max_val_acc :
            print(f'higher valid acc obtained: {val_acc * 100}')
            max_val_acc = val_acc * 100
            torch.save(student.state_dict(), savename)

    return student, train_loss, val_loss, val_acc, max_val_acc
