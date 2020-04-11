cd ../datasets

python3 create_dataset.py -d imagenette -p 10 -s 1
python3 create_dataset.py -d imagenette -p 20 -s 1
python3 create_dataset.py -d imagenette -p 30 -s 1
python3 create_dataset.py -d imagenette -p 40 -s 1

python3 create_dataset.py -d imagewoof -p 10 -s 1
python3 create_dataset.py -d imagewoof -p 20 -s 1
python3 create_dataset.py -d imagewoof -p 30 -s 1
python3 create_dataset.py -d imagewoof -p 40 -s 1

python3 create_dataset.py -d cifar10 -p 10 -s 1
python3 create_dataset.py -d cifar10 -p 20 -s 1
python3 create_dataset.py -d cifar10 -p 30 -s 1
python3 create_dataset.py -d cifar10 -p 40 -s 1

cd ~/.fastai/data/imagenette/models
wget https://www.dropbox.com/s/p8zeahm8f6ehkfy/resnet34_imagenette_bs64.pth

cd ~/.fastai/data/imagewoof/models
wget https://www.dropbox.com/s/9xoj7d4qrlf94pc/resnet34_imagewoof_bs64.pth

cd ~