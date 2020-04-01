import numpy as np


def get_savename(hyper_params, mode=None, p=None):
    if p is not None:
        less = '/less_data/' + '_' + str(p) + '_'
    else:
        less = ''
    if mode == 'stagewise':
        return '../saved_models/' + less + hyper_params['model'] + '/stage' + str(hyper_params['stage'] - 1) + '/model' + str(
            hyper_params['seed']) + '.pt'
    if mode == 'classifier':
        return '../saved_models/' + less + hyper_params['model'] + '/classifier/model' + str(hyper_params['seed']) + '.pt'


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class encode_segmap(object):
    def __init__(self, ignore_index):
        self.void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_labels, range(19)))
        self.ignore_index = ignore_index

    def __call__(self, mask):
        mask = np.array(mask)
        for voidc in self.void_labels:
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels:
            mask[mask == validc] = self.class_map[validc]
        return mask


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


class SaveFeatures:
    def __init__(self, m):
        self.handle = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, m, inp, outp):
        self.features = outp

    def remove(self):
        self.handle.remove()


def get_features_trad(student, teacher):
    sf_student = [SaveFeatures(m) for m in [student.encoder.layer2,
                                            student.decoder.blocks[2],
                                            ]]

    sf_teacher = [SaveFeatures(m) for m in [teacher.encoder.layer2,
                                            teacher.decoder.blocks[2],
                                            ]]
    return sf_student, sf_teacher


def get_features(student, teacher):
    sf_student = [SaveFeatures(m) for m in [student.encoder.relu,
                                            student.encoder.layer1,
                                            student.encoder.layer2,
                                            student.encoder.layer3,
                                            student.encoder.layer4,
                                            student.decoder.blocks[0],
                                            student.decoder.blocks[1],
                                            student.decoder.blocks[2],
                                            student.decoder.blocks[3],
                                            student.decoder.blocks[4]
                                            ]]

    sf_teacher = [SaveFeatures(m) for m in [teacher.encoder.relu,
                                            teacher.encoder.layer1,
                                            teacher.encoder.layer2,
                                            teacher.encoder.layer3,
                                            teacher.encoder.layer4,
                                            teacher.decoder.blocks[0],
                                            teacher.decoder.blocks[1],
                                            teacher.decoder.blocks[2],
                                            teacher.decoder.blocks[3],
                                            teacher.decoder.blocks[4]
                                            ]]
    return sf_student, sf_teacher
