#!/usr/bin/env python
# coding: utf-8

# In[30]:


import os
from pathlib import Path
from fastai.vision import *
import shutil
import random
NUM_ = 300

data = untar_data(URLs.IMAGENETTE)

new_data = data/"new"
new_data.mkdir(exist_ok=True)

try:test = shutil.copytree(data/"train", new_data/"test")
except: test = new_data/"test"
    
try: val = shutil.copytree(data/"val", new_data/"val")
except: val = new_data/"val"

train = new_data/"train"
shutil.rmtree(str(train))
train.mkdir()

for i in test.ls():
    (train/i.stem).mkdir(exist_ok=True) 
    
for folder in test.ls():
    ls = folder.ls()
    sample = random.sample(ls, NUM_)
    for file in sample:
        shutil.copy(file, train/folder.stem/file.stem)

def test_folder(folder_name, NUM_):
    for i in folder_name.ls():
        assert len(i.ls()) == NUM_
    else:
        return True

if test_folder(train, NUM_): print("Completed copying")



