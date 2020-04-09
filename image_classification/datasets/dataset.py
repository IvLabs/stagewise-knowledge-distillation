from fastai.vision import *

def get_dataset(dataset, percentage=None, test=False):
    val = 'val'
    sz = 224
    stats = imagenet_stats
    if dataset == 'imagenette' : 
        path = untar_data(URLs.IMAGENETTE)
    elif dataset == 'cifar10' : 
        path = untar_data(URLs.CIFAR)
    elif dataset == 'imagewoof' : 
        path = untar_data(URLs.IMAGEWOOF)
    else:
        sys.exit(f'invalid dataset : {dataset}')
    
    tfms = get_transforms(do_flip=False)
    if hyper_params['dataset'] == 'cifar10' : 
        val = 'test'
        sz = 32
        stats = cifar_stats

    return ImageDataBunch.from_folder(path, train = 'train', valid = val, bs = hyper_params["batch_size"], size = sz, ds_tfms = tfms).normalize(stats)