from fastai.vision import *

def get_dataset(dataset, batch_size, percentage=None):
    val = 'val'
    sz = 224
    stats = imagenet_stats
    if dataset == 'imagenette' : 
        path = untar_data('https://s3.amazonaws.com/fast-ai-imageclas/imagenette')
        classes = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
    elif dataset == 'cifar10' : 
        path = untar_data(URLs.CIFAR)
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset == 'imagewoof' : 
        path = untar_data('https://s3.amazonaws.com/fast-ai-imageclas/imagewoof')
        classes = ['n02086240', 'n02087394', 'n02088364', 'n02089973', 'n02093754', 'n02096294', 'n02099601', 'n02105641', 'n02111889', 'n02115641']
    else:
        sys.exit(f'invalid dataset : {dataset}')
    
    if percentage is not None:
        path = path/('new' + str(percentage))

    tfms = get_transforms(do_flip=False)
    if dataset == 'cifar10' : 
        val = 'test'
        sz = 32
        stats = cifar_stats

    return ImageDataBunch.from_folder(path, train='train', valid=val, bs=batch_size, size=sz, ds_tfms=tfms, classes=classes).normalize(stats)
