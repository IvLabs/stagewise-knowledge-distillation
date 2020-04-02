import argparse


def get_args(desc="No Desc", mode='train'):
    if mode == 'train':
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('-m','--model', choices=['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26', 'resnet34'],
                            help='Give the encoder name from the choices', default="resnet26")
        parser.add_argument('-d', '--dataset', choices=['camvid', 'cityscapes'],
                            help='Give the dataset to be used for training from the choices', default="camvid")
        parser.add_argument('-e', '--epoch', type=int, help='Give number of epochs for training', default=10)
        parser.add_argument('-s', '--seed', type=int, help='Give the random seed number')
        parser.add_argument('-p', '--percentage', type=int, help='Percentage of dataset to be used for training', default=None)
        parser.add_argument('-g', '--gpu', choices=[0, 1, 'cpu'], help='which gpu (or cpu)', default=0)
        args = parser.parse_args()

    elif mode == 'eval':
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('-d', '--dataset', choices=['camvid', 'cityscapes'],
                            help='Give the dataset to be used for training from the choices', default="camvid")
        parser.add_argument('-s', '--seed', type=int, help='Give the random seed number')
        parser.add_argument('-g', '--gpu', choices=[0, 1, 'cpu'], help='which gpu (or cpu)', default=0)
        args = parser.parse_args()

    elif mode == 'data':
        parser = argparse.ArgumentParser(description='Reducing dataset size')
        parser.add_argument('-p', '--percentage', type=int, help='Give percentage of dataset')
        args = parser.parse_args()
    else: print(f'invalid mode: {mode}')
    return args
