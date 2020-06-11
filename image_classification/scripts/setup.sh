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

mkdir -p ~/.fastai/data/imagenette/models
cd ~/.fastai/data/imagenette/models
wget https://www.dropbox.com/s/p8zeahm8f6ehkfy/resnet34_imagenette_bs64.pth

mkdir -p ~/.fastai/data/imagewoof/models
cd ~/.fastai/data/imagewoof/models
wget https://www.dropbox.com/s/9xoj7d4qrlf94pc/resnet34_imagewoof_bs64.pth

mkdir -p ~/.fastai/data/cifar10/models
cd ~/.fastai/data/cifar10/models
wget https://www.dropbox.com/s/qgay9te38hkimbb/resnet34_cifar_bs64.pth

cd ~
