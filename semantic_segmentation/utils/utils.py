from torchvision import transforms
import os
import numpy as np


def get_savename(hyper_params, dataset='camvid', mode=None, p=None):
    assert mode in ['stagewise', 'classifier', 'traditional-kd', 'traditional-stage', 'simultaneous', 'pretrain']

    if p is not None:
        less = f'less_data{str(p)}'
    else:
        less = 'full_data'

    if mode == 'stagewise':
        savename = f'../saved_models/{dataset}/{less}/{hyper_params["model"]}/stage{str(hyper_params["stage"])}'
    elif mode == 'traditional-stage':
        savename = f'../saved_models/{dataset}/{less}/{hyper_params["model"]}/traditional-kd/stage{str(hyper_params["stage"])}'
    else:
        savename = f'../saved_models/{dataset}/{less}/{hyper_params["model"]}/{mode}'

    os.makedirs(savename, exist_ok=True)
    return savename + f'/model{str(hyper_params["seed"])}.pt'


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def get_tf():
    return transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize(mean=[0.41189489566336, 0.4251328133025, 0.4326707089857],
                            std=[0.27413549931506, 0.28506257482912, 0.28284674400252])])


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
