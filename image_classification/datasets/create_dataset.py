import os
from pathlib import Path
from fastai.vision import *
import shutil
import random
import argparse
import sys
from image_classification.arguments import get_args

# parser = argparse.ArgumentParser(description = 'Creating dataset for less data')
# parser.add_argument('-d', choices = ['imagenette', 'imagewoof', 'cifar10'], help = 'Give the dataset name from the choices')
# parser.add_argument('-p', type = int, help = 'Give percentage of dataset')
# args = parser.parse_args()

args = get_args(description='Creating dataset for less data experiments', mode='data')
random.seed(args.seed)

if args.dataset == 'imagenette' or args.dataset == 'imagewoof' :
    NUM_ = int(args.percentage * 13)
    if args.dataset == 'imagenette' :
        data = untar_data(URLs.IMAGENETTE)
    else :
        data = untar_data(URLs.IMAGEWOOF)
elif args.dataset == 'cifar10' :
    NUM_ = int(args.percentage * 50)
    data = untar_data(URLs.CIFAR)
else :
    print('Give dataset from choices only')

new_data = data/('new' + str(args.percentage))
new_data.mkdir(exist_ok = True)

try : 
    test = shutil.copytree(data/"train", new_data/"test")
except : 
    test = new_data/"test"
    
try :
    if args.dataset == 'imagenette' or args.dataset == 'imagewoof' :
        val = shutil.copytree(data/"val", new_data/"val")
    else :
        val = shutil.copytree(data/"test", new_data/"val")
except :
    val = new_data/"val"

train = new_data/"train"
try :
    shutil.rmtree(str(train)) 
except : 
    pass
train.mkdir()

for i in test.ls() :
    (train/i.stem).mkdir(exist_ok = True) 
    
for folder in test.ls() :
    ls = folder.ls()
    sample = random.sample(ls, NUM_)
    for file in sample :
        shutil.move(file, str(train/folder.stem/file.stem) + ".JPEG")

def test_folder(folder_name, NUM_):
    for i in folder_name.ls():
        assert len(i.ls()) == NUM_
    else:
        return True

if test_folder(train, NUM_): 
    print('Completed copying', args.percentage, '% of', args.dataset)
