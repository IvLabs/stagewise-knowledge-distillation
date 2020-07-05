cd ..
git clone https://github.com/alexgkendall/SegNet-Tutorial.git
mv SegNet-Tutorial data
 # 
mkdir -p saved_models/resnet34
cd saved_models/resnet34
wget https://www.dropbox.com/s/hgi85xq7p2khycm/pretrained_0.pt
# 
cd ../..

python3 utils/split_camvid.py -p 10
python3 utils/split_camvid.py -p 20
python3 utils/split_camvid.py -p 30
python3 utils/split_camvid.py -p 40

cd ../..
