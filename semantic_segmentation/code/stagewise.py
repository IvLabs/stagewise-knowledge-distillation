import torchvision

from trainer import train_stagewise, unfreeze, train_classifer
from dataset import CamVid, Cityscapes
from core import *
from args import get_args

torch.cuda.set_device(0)

args = get_args(desc='Stagewise training of UNet based on ResNet encoder')

hyper_params = {
    "model": args.m,
    "seed": args.s,
    "num_classes": 12,
    "batch_size": 8,
    "num_epochs": args.e,
    "learning_rate": 1e-4,
    "stage": 0
}

torch.manual_seed(args.s)
torch.cuda.manual_seed(args.s)

if args.d == 'camvid':
    num_classes = 12
    DATA_DIR = '../data/CamVid/'
    train_dataset = CamVid(DATA_DIR, mode='train', p=args.p)
    valid_dataset = CamVid(DATA_DIR, mode='val', p=args.p)

elif args.d == 'cityscapes' :
    num_classes = 19
    root = '/home/himanshu/cityscape/'
    target_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((480, 640)), encode_segmap(ignore_index = 250)])
    tfsm = torchvision.transforms.Compose([torchvision.transforms.Resize((480, 640)), torchvision.transforms.ToTensor()])
    train_dataset = Cityscapes(root=root,
                                folder = 'leftImg8bit',
                                split = 'train',
                                target_type='semantic',
                                mode='fine',
                                transform = tfsm,
                                target_transform = target_transform
                                )
    valid_dataset = Cityscapes(root=root,
                                folder = 'leftImg8bit',
                                split = 'val',
                                target_type='semantic',
                                mode='fine',
                                transform = tfsm,
                                target_transform = target_transform
                                )


trainloader = DataLoader(train_dataset, batch_size=hyper_params['batch_size'], shuffle=True, drop_last=True)
valloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

student = models.unet.Unet(hyper_params['model'], classes=12, encoder_weights=None).cuda()
teacher = models.unet.Unet('resnet34', classes=12, encoder_weights=None).cuda()
teacher.load_state_dict(torch.load('../saved_models/resnet34/pretrained_0.pt'))
# Freeze the teacher model
teacher = unfreeze(teacher, 30)

sf_student, sf_teacher = get_features(student, teacher)

_, student, hyper_params = train_stagewise(hyper_params, teacher, student, sf_teacher,
                                           sf_student, trainloader, valloader)
# Classifier training
hyper_params['stage'] = 10
student.load_state_dict(torch.load(get_savename(hyper_params, mode='stagewise')))

train_classifer(hyper_params, student, trainloader, valloader)
