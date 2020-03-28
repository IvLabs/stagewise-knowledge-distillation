import torch
from torch.utils import data
from torchvision import transforms
import torchvision
import zipfile
from torchvision.datasets.utils import *
import torch.nn.functional as F
import numpy as np
import os
import cv2
from collections import namedtuple
from tqdm import tqdm
from PIL import Image
   
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

class encode_segmap(object) :
    def __init__(self, ignore_index) :
        self.void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_map = dict(zip(self.valid_labels, range(19)))
        self.ignore_index = ignore_index
    def __call__(self, mask) : 
        mask = np.array(mask)
        for voidc in self.void_labels :
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels :
            mask[mask == validc] = self.class_map[validc]
        return mask

def iou(mask1, mask2, num_classes = 19, smooth = 1e-6) :
    avg_iou = 0
    for sem_class in range(num_classes) : 
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection_now = (pred_inds[target_inds]).long().sum().item()
        union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
        avg_iou += (float(intersection_now + smooth) / float(union_now + smooth))
    return(avg_iou / num_classes)
    
def dice_coeff(mask1, mask2, smooth = 1e-6, num_classes = 19) : 
    dice = 0
    for sem_class in range(num_classes) : 
        pred_inds = (mask2 == sem_class)
        target_inds = (mask1 == sem_class)
        intersection = (pred_inds[target_inds]).long().sum().item()
        denom = pred_inds.long().sum().item() + target_inds.long().sum().item()
        dice += (float(2 * intersection) + smooth) / (float(denom) + smooth)
    return dice / num_classes
    
def pixelwise_acc(mask1, mask2) :
    equals = (mask1 == mask2).sum().item()
    return equals / (mask1.shape[0] * mask1.shape[1] * mask1.shape[2])
    
class KITTI(data.Dataset):
    def __init__(self, split = 'test', transform = None):
        self.void_labels = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_labels = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(19)))
        self.split = split
        self.img_path = '../testing/image_2/'
        self.mask_path = None
        if self.split == 'train':
            self.img_path = '../training/image_2/'    
            self.mask_path = '../training/semantic/'
        self.transform = transform
        
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = None
        if self.split == 'train':
            self.mask_list = self.get_filenames(self.mask_path)
        
    def __len__(self):
        return(len(self.img_list))
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.resize(img, (1024, 256))
        mask = None
        if self.split == 'train':
            mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (1024, 256))
            mask = self.encode_segmap(mask)
            assert(mask.shape == (256, 1024))
        
        if self.transform:
            img = self.transform(img)
            assert(img.shape == (3, 256, 1024))
        else :
            assert(img.shape == (256, 1024, 3))
        
        if self.split == 'train':
            return img, mask
        else :
            return img
    
    def encode_segmap(self, mask):
        '''
        Sets void classes to ignore_index so they won't be considered for training
        '''
        for voidc in self.void_labels :
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels :
            mask[mask == validc] = self.class_map[validc]
        return mask
    
    def get_filenames(self, path):
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list

class CamVid(data.Dataset):
    """CamVid Dataset. Read images, apply transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask    
    """
    
    CLASSES = ['sky', 'building', 'pole', 'road', 'pavement', 
               'tree', 'signsymbol', 'fence', 'car', 
               'pedestrian', 'bicyclist', 'unlabelled']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes = None, 
            transform = None
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.transform = transform
        
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.resize(image, (640, 480))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (640, 480))
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        mask = np.argmax(mask, axis = 2)
        
        assert(mask.shape == (480, 640))
        assert(image.shape == (480, 640, 3))
        
        if self.transform :
            image = self.transform(image)
            assert(image.shape == (3, 480, 640))
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

class Cityscapes(torchvision.datasets.vision.VisionDataset):
    """
    `Cityscapes <http://www.cityscapes-dataset.com/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory ``leftImg8bit``
            and ``gtFine`` or ``gtCoarse`` are located.
        folder (string): directory of images to be used
        split (string, optional): The image split to use, ``train``, ``test`` or ``val`` if mode="gtFine"
            otherwise ``train``, ``train_extra`` or ``val``
        mode (string, optional): gtFine for full dataset or folder name of targets
        target_type (string or list, optional): Type of target to use, ``instance``, ``semantic``, ``polygon``
            or ``color``. Can also be a list to output a tuple with all specified target types.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    # Based on https://github.com/mcordts/cityscapesScripts
    CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                     'has_instances', 'ignore_in_eval', 'color'])

    classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

    def __init__(self, root, folder, split='train', mode='fine', target_type='instance',
                 transform=None, target_transform=None, transforms=None):
        super(Cityscapes, self).__init__(root, transforms, transform, target_transform)
        self.mode = 'gtFine' if mode == 'fine' else mode
        self.images_dir = os.path.join(self.root, folder, split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        if mode == "fine":
            valid_modes = ("train", "test", "val")
        else:
            valid_modes = ("train", "train_extra", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        [verify_str_arg(value, "target_type",
                        ("instance", "semantic", "polygon", "color"))
         for value in self.target_type]

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format('_trainvaltest.zip'))

            if self.mode == 'gtFine':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '_trainvaltest.zip'))
            elif self.mode == 'gtCoarse':
                target_dir_zip = os.path.join(self.root, '{}{}'.format(self.mode, '.zip'))

            if os.path.isfile(image_dir_zip) and os.path.isfile(target_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
                extract_archive(from_path=target_dir_zip, to_path=self.root)
            else:
                raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                                   ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = '{}_{}'.format(file_name.split('_leftImg8bit')[0],
                                                 self._get_target_suffix(self.mode, t))
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        targets = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}", "Type: {target_type}"]
        return '\n'.join(lines).format(**self.__dict__)

    def _load_json(self, path):
        with open(path, 'r') as file:
            data = json.load(file)
        return data

    def _get_target_suffix(self, mode, target_type):
        if target_type == 'instance':
            return '{}_instanceIds.png'.format(mode)
        elif target_type == 'semantic':
            return '{}_labelIds.png'.format(mode)
        elif target_type == 'color':
            return '{}_color.png'.format(mode)
        else:
            return '{}_polygons.json'.format(mode)

def train(model, train_loader, val_loader, num_classes, epoch, num_epochs, loss_function, optimiser, scheduler, savename, highest_iou):
    model.train()
    losses = list()
    val_losses = list()
    gpu1 = 'cuda:0'
    ious = list()
    pixel_accs = list()
    dices = list()
    max_iou = highest_iou
    savename2 = savename[ : -3] + '_opt.pt'
    loop = tqdm(train_loader)
    num_steps = len(loop)
    for data, target in loop:
        model.train()
        model = model.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)
        
        optimiser.zero_grad()
        prediction = model(data)
        loss = loss_function(prediction, target)
        prediction = F.softmax(prediction, dim = 1)
        prediction = torch.argmax(prediction, axis = 1).squeeze(1)
        
        losses.append(loss.item())
        
        loss.backward()
        optimiser.step()
        scheduler.step()
        
        loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        loop.set_postfix(loss = loss.item())

    model.eval()
    for data, target in val_loader : 
        model = model.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        prediction = model(data)
        val_loss = loss_function(prediction, target)
        prediction = F.softmax(prediction, dim = 1)
        prediction = torch.argmax(prediction, axis = 1).squeeze(1)

        ious.append(iou(target, prediction, num_classes = num_classes))
        dices.append(dice_coeff(target, prediction, num_classes = num_classes))
        pixel_accs.append(pixelwise_acc(prediction, target))
        val_losses.append(val_loss.item())
    
    avg_iou = sum(ious) / len(ious)
    avg_dice_coeff = sum(dices) / len(dices)
    avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)

    if avg_iou > max_iou :
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

class SaveFeatures :
    def __init__(self, m) : 
        self.handle = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, m, inp, outp) : 
        self.features = outp
    def remove(self) :
        self.handle.remove()

def unfreeze_trad(model, stage) : 
    # First asymmetrical stage
    if stage == 0 :
        for name, param in model.named_parameters() :
            param.requires_grad = False
            if name.startswith('encoder.conv') or name.startswith('encoder.bn') or name.startswith('encoder.layer0') or name.startswith('encoder.layer1') or name.startswith('encoder.layer2') :
                param.requires_grad = True
    elif stage == 1 : 
        for name, param in model.named_parameters() :
            param.requires_grad = False
            if name.startswith('encoder.layer3') or name.startswith('encoder.layer4') or name.startswith('decoder.blocks.0') or name.startswith('decoder.blocks.1') or name.startswith('decoder.blocks.2') :
                param.requires_grad = True
    # Classifier stage
    elif stage == 2 : 
        for name, param in model.named_parameters() :
            param.requires_grad = False
            if name.startswith('segmentation') or name.startswith('decoder.blocks.3') or name.startswith('decoder.blocks.4') :
                param.requires_grad = True
    # Freeze everything
    else :
        for name, param in model.named_parameters() : 
            param.requires_grad = False
    
    return model

def unfreeze(model, stage) : 
    # First asymmetrical stage
    if stage == 0 :
        for name, param in model.named_parameters() :
            param.requires_grad = False
            if name.startswith('encoder.conv') or name.startswith('encoder.bn') : 
                param.requires_grad = True
    # Encoder stages
    elif stage > 0 and stage < 5 : 
        for name, param in model.named_parameters() :
            param.requires_grad = False
            if name.startswith('encoder.layer' + str(stage)) : 
                param.requires_grad = True
    # Decoder stages
    elif stage > 4 and stage < 10 : 
        for name, param in model.named_parameters() :
            param.requires_grad = False
            if name.startswith('decoder.blocks.' + str(stage - 5)) : 
                param.requires_grad = True
    # Classifier stage
    elif stage == 10 : 
        for name, param in model.named_parameters() :
            param.requires_grad = False
            if name.startswith('segmentation') :
                param.requires_grad = True
    # Freeze everything
    else :
        for name, param in model.named_parameters() : 
            param.requires_grad = False
    
    return model

def train_stage(model, teacher, stage, sf_student, sf_teacher, train_loader, val_loader, epoch, num_epochs, loss_function, optimiser, scheduler, savename, lowest_val):
    model.train()
    losses = list()
    val_losses = list()
    gpu1 = 'cuda:0'
    ious = list()
    pixel_accs = list()
    dices = list()
    lowest_val_loss = lowest_val
    savename2 = savename[ : -3] + '_opt.pt'
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
        loop.set_postfix(loss = loss.item())

    model.eval()
    for data, target in val_loader : 
        model = model.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        prediction = model(data)
        _ = teacher(data)
        val_loss = loss_function(sf_student[stage].features, sf_teacher[stage].features)
        
        val_losses.append(val_loss.item())
    
    avg_loss = sum(losses) / len(losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    if avg_val_loss < lowest_val_loss :
        lowest_val_loss = avg_val_loss
        torch.save(model.state_dict(), savename)
        torch.save(optimiser.state_dict(), savename2)
        print('new lowest_val_loss', lowest_val_loss)

    return model, lowest_val_loss, avg_loss, avg_val_loss

def train_trad_kd(model, teacher, sf_teacher, sf_student, train_loader, val_loader, num_classes, epoch, num_epochs, loss_function1, loss_function2, optimiser, scheduler, savename, highest_iou):
    model.train()
    losses = list()
    val_losses = list()
    gpu1 = 'cuda:0'
    ious = list()
    pixel_accs = list()
    dices = list()
    max_iou = highest_iou
    savename2 = savename[ : -3] + '_opt.pt'
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

        loss = loss_function1(prediction, target) + loss_function2(sf_teacher[0].features, sf_student[0].features) + loss_function2(sf_teacher[1].features, sf_student[1].features)
        prediction = F.softmax(prediction, dim = 1)
        prediction = torch.argmax(prediction, axis = 1).squeeze(1)
        
        losses.append(loss.item())
        
        loss.backward()
        optimiser.step()
        scheduler.step()
        
        loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        loop.set_postfix(loss = loss.item())

    model.eval()
    for data, target in val_loader : 
        model = model.to(gpu1)
        teacher = teacher.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        prediction = model(data)
        _ = teacher(data)
        val_loss = loss_function1(prediction, target) + loss_function2(sf_teacher[0].features, sf_student[0].features) + loss_function2(sf_teacher[1].features, sf_student[1].features)
        prediction = F.softmax(prediction, dim = 1)
        prediction = torch.argmax(prediction, axis = 1).squeeze(1)

        ious.append(iou(target, prediction, num_classes = num_classes))
        dices.append(dice_coeff(target, prediction, num_classes = num_classes))
        pixel_accs.append(pixelwise_acc(prediction, target))
        val_losses.append(val_loss.item())
    
    avg_iou = sum(ious) / len(ious)
    avg_dice_coeff = sum(dices) / len(dices)
    avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)

    if avg_iou > max_iou :
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

def train_simult(model, teacher, sf_teacher, sf_student, train_loader, val_loader, num_classes, epoch, num_epochs, loss_function, loss_function2, optimiser, scheduler, savename, highest_iou):
    model.train()
    losses = list()
    val_losses = list()
    gpu1 = 'cuda:0'
    ious = list()
    pixel_accs = list()
    dices = list()
    max_iou = highest_iou
    savename2 = savename[ : -3] + '_opt.pt'
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
        for i in range(10) :
            loss += loss_function2(sf_teacher[i].features, sf_student[i].features)
        loss += loss_function(prediction, target)
        loss /= 11
        prediction = F.softmax(prediction, dim = 1)
        prediction = torch.argmax(prediction, axis = 1).squeeze(1)
        
        losses.append(loss.item())
        
        loss.backward()
        optimiser.step()
        scheduler.step()
        
        loop.set_description('Epoch {}/{}'.format(epoch + 1, num_epochs))
        loop.set_postfix(loss = loss.item())

    model.eval()
    for data, target in val_loader : 
        model = model.to(gpu1)
        data, target = data.float().to(gpu1), target.long().to(gpu1)

        prediction = model(data)
        val_loss = loss_function(prediction, target)
        prediction = F.softmax(prediction, dim = 1)
        prediction = torch.argmax(prediction, axis = 1).squeeze(1)

        ious.append(iou(target, prediction, num_classes = num_classes))
        dices.append(dice_coeff(target, prediction, num_classes = num_classes))
        pixel_accs.append(pixelwise_acc(prediction, target))
        val_losses.append(val_loss.item())
    
    avg_iou = sum(ious) / len(ious)
    avg_dice_coeff = sum(dices) / len(dices)
    avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)

    if avg_iou > max_iou :
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
