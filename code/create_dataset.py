import os
from pathlib import Path
from fastai.vision import *
import shutil
import random
NUM_ = 1250

random.seed(1)

data = untar_data(URLs.CIFAR)

new_data = data/"new"
new_data.mkdir(exist_ok = True)

try : 
    test = shutil.copytree(data/"train", new_data/"test")
except : 
    test = new_data/"test"
    
try : 
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
    print("Completed copying")