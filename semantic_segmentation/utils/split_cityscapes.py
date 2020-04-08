import os
import random
import sys
from pathlib import Path
from shutil import copyfile

from arguments import get_args

random.seed(1)

args = get_args('args for datasplit (citiscapes)', mode='data')

current_path = os.path.abspath('')
train_path = "/home/himanshu/cityscape/leftImg8bit/train"
train_path_lb = "/home/himanshu/cityscape/gtFine/train"
perc = args.percentage

if perc < 0 or perc > 100:
    print('Illegal usage of -p, only between 0 and 100')
    sys.exit(0)

destination = Path("/home/akshay/cityscape/frac" + str(perc) + "/leftImg8bit/train")
os.makedirs(str(destination), exist_ok=True)

destination_lb = Path("/home/akshay/cityscape/frac" + str(perc) + "/gtFine/train")
os.makedirs(str(destination_lb), exist_ok=True)

for folder in os.listdir(train_path):
    des1 = Path(destination / folder)
    des1.mkdir(exist_ok=True)

    des1_lb = Path(destination_lb / folder)
    des1_lb.mkdir(exist_ok=True)

    filels = os.listdir(f'{train_path}/{folder}')
    fracfiles = random.sample(filels, int(len(filels) * perc // 100))

    for files in fracfiles:
        copyfile(f'{train_path}/{folder}/{files}', f'{str(des1)}/{files}')
        copyfile(f'{train_path_lb}/{folder}/{files[:-15]}gtFine_labelIds.png',
                 f'{str(des1_lb)}/{files[:-15]}gtFine_labelIds.png')
