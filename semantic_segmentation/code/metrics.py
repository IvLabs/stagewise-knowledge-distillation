import torch
import torch.functional as F


def iou(mask1, mask2, num_classes=19, smooth=1e-6):
    avg_iou = 0
    for sem_class in range(num_classes):
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection_now = (pred_inds[target_inds]).long().sum().item()
        union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
        avg_iou += (float(intersection_now + smooth) / float(union_now + smooth))
    return (avg_iou / num_classes)


def dice_coeff(mask1, mask2, smooth=1e-6, num_classes=19):
    dice = 0
    for sem_class in range(num_classes):
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection = (pred_inds[target_inds]).long().sum().item()
        denom = pred_inds.long().sum().item() + target_inds.long().sum().item()
        dice += (float(2 * intersection) + smooth) / (float(denom) + smooth)
    return dice / num_classes


def pixelwise_acc(mask1, mask2):
    equals = (mask1 == mask2).sum().item()
    return equals / (mask1.shape[0] * mask1.shape[1] * mask1.shape[2])


def mean_iou(model, dataloader, args):
    gpu1 = args.gpu
    ious = list()
    for i, (data, target) in enumerate(dataloader):
        data, target = data.float().to(gpu1), target.long().to(gpu1)
        prediction = model(data)
        prediction = F.softmax(prediction, dim=1)
        prediction = torch.argmax(prediction, axis=1).squeeze(1)
        ious.append(iou(target, prediction, num_classes=11))

    return (sum(ious) / len(ious))
