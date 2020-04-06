import os
import random
import shutil
import sys

from arguments import get_args

random.seed(1)

args = get_args('args for datasplit (camvid)', mode='data')

current_path = os.path.abspath('')

perc = args.percentage

if not 0 < perc < 100:
    print('Illegal usage of -p, only between 0 and 100')
    sys.exit(0)

NUM = int(367 * perc / 100)

image_dir = os.path.join(current_path, 'data/CamVid/train')
mask_dir = os.path.join(current_path, 'data/CamVid/trainannot')

new_dir = os.path.join(current_path, 'data/CamVid/trainsmall' + str(perc))
new_mask_dir = os.path.join(current_path, 'data/CamVid/trainsmallannot' + str(perc))

os.makedirs(new_dir, exist_ok=True)
os.makedirs(new_mask_dir, exist_ok=True)
ls = os.listdir(image_dir)
sample = random.sample(ls, NUM)
for file in sample:
    shutil.copy(os.path.join(image_dir, file), os.path.join(new_dir, file))
    shutil.copy(os.path.join(mask_dir, file), os.path.join(new_mask_dir, file))
