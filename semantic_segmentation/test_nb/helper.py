import torch
from torch.utils import data
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import os
import cv2
from tqdm import tqdm
   
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
    
def train(model, train_loader, val_loader, epoch, num_epochs, loss_function, optimiser, scheduler, savename, highest_iou):
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

        ious.append(iou(target, prediction, num_classes = 11))
        dices.append(dice_coeff(target, prediction, num_classes = 11))
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
        prediction = F.softmax(prediction, dim = 1)
        prediction = torch.argmax(prediction, axis = 1).squeeze(1)

        ious.append(iou(target, prediction, num_classes = 11))
        dices.append(dice_coeff(target, prediction, num_classes = 11))
        pixel_accs.append(pixelwise_acc(prediction, target))
        val_losses.append(val_loss.item())
    
    avg_iou = sum(ious) / len(ious)
    avg_dice_coeff = sum(dices) / len(dices)
    avg_pixel_acc = sum(pixel_accs) / len(pixel_accs)
    
    print('avg_iou: ', avg_iou)
    print('avg_pixel_acc: ', avg_pixel_acc)
    print('avg_dice_coeff: ', avg_dice_coeff)
    
    avg_loss = sum(losses) / len(losses)
    avg_val_loss = sum(val_losses) / len(val_losses)
    if avg_val_loss < lowest_val_loss :
        lowest_val_loss = avg_val_loss
        torch.save(model.state_dict(), savename)
        torch.save(optimiser.state_dict(), savename2)
        print('new lowest_val_loss', lowest_val_loss)

    return model, lowest_val_loss, avg_loss, avg_val_loss, avg_iou, avg_pixel_acc, avg_dice_coeff