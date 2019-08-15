from fastai.vision import *
import torch
torch.cuda.set_device(1)

path = untar_data(URLs.CIFAR)

data = ImageDataBunch.from_folder(path, train = 'train', valid = 'test', bs = 64)
data.normalize(cifar_stats)

learn = cnn_learner(data, models.resnet34, metrics = accuracy)

learn.fit(100)

learn.export(file = 'bs64.pkl')



