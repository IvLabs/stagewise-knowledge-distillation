import os
import random 
import shutil 
import sys
import argparse

random.seed(1)

parser = argparse.ArgumentParser(description = 'Reducing dataset size')
parser.add_argument('-p', type = int, help = 'Give percentage of dataset')
args = parser.parse_args()

current_path = os.path.abspath('..')

perc = args.p

if perc < 0 or perc > 100 : 
    print('Illegal usage of -p, only between 0 and 100')
    sys.exit(0)

NUM = int(367 * perc / 100)

image_dir = os.path.join(current_path, 'data/CamVid/train')
mask_dir = os.path.join(current_path, 'data/CamVid/trainannot')

new_dir = os.path.join(current_path, 'data/CamVid/trainsmall' + str(perc))
new_mask_dir = os.path.join(current_path, 'data/CamVid/trainsmallannot' + str(perc))

os.makedirs(new_dir, exist_ok = True)
os.makedirs(new_mask_dir, exist_ok = True)
ls = os.listdir(image_dir)
sample = random.sample(ls, NUM)
for file in sample : 
    shutil.copy(os.path.join(image_dir, file), os.path.join(new_dir, file))
    shutil.copy(os.path.join(mask_dir, file), os.path.join(new_mask_dir, file))