import argparse


def get_args(desc="No Desc", small=False, data=False):
    if not data:
        parser = argparse.ArgumentParser(description=desc)
        parser.add_argument('-m', choices=['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26', 'resnet34'],
                            help='Give the encoder name from the choices', default="resnet26")
        parser.add_argument('-d', choices=['camvid', 'cityscapes'],
                            help='Give the dataset to be used for training from the choices', default="cityscapes")
        parser.add_argument('-e', type=int, help='Give number of epochs for training')
        parser.add_argument('-s', type=int, help='Give the random seed number')
        if small:
            parser.add_argument('-p', type=int, help='Percentage of dataset to be used for training')
        args = parser.parse_args()
    else:
        parser = argparse.ArgumentParser(description='Reducing dataset size')
        parser.add_argument('-p', type=int, help='Give percentage of dataset')
        args = parser.parse_args()
    return args

