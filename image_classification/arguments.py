import sys
import argparse

def get_args(description, mode='train'):
    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('-s', '--seed', type=int, help = 'Give random seed')
    parser.add_argument('-d', '--dataset', choices = ['imagenette', 'imagewoof', 'cifar10'], help = 'Give the dataset name from the choices')
    if mode == 'train':
        parser.add_argument('-m', '--model', choices = ['resnet10', 'resnet14', 'resnet18', 'resnet20', 'resnet26'], help = 'Give the model name from the choices')
        parser.add_argument('-e', '--epoch', type=int, help = 'Give number of epochs for training')
        parser.add_argument('-p', '--percentage', type=int, help='Percentage of dataset to be used for training', default=None)
        parser.add_argument('-g', '--gpu', choices=[0, 1, 'cpu'], help='which GPU (or CPU) to be used for training', default=0)

    elif mode == 'eval':
        parser.add_argument('-g', '--gpu', choices=[0, 1, 'cpu'], help='which GPU (or CPU) to be used for training', default=0)

    elif mode == 'data':
        parser.add_argument('-p', '--percentage', type=int, help='Give percentage of dataset')

    else: 
        sys.exit(f'invalid mode: {mode}')

    args = parser.parse_args()
    return args