# Imports packages
import argparse

# Create command line argument parser
def get_input_args_train():
    parser = argparse.ArgumentParser(
        description='Train a neural network on a data set and save the model to a checkpoint file')
    # get directory of images folder
    parser.add_argument('dir', type=str, default='flowers',
                        help='path to the folder of images dataset')
    # get directory to save checkpoint file
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='set directory to save checkpoint file')
    # get CNN model architecture
    parser.add_argument('--arch', type=str, default='vgg16', choices=['alexnet', 'densenet', 'vgg16'],
                        help="CNN model architecture: densenet, alexnet, or vgg16")
    # get learning rate
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help='set learning rate')
    # get size of hidden layer for classifier
    parser.add_argument('-hu', '--hidden_units', type=int, default=512,
                        help='set number of hidden units')
    # get size number of epoches
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='set number of epoches')
    # select gpu as computation device
    parser.add_argument('--gpu', action='store_true',
                        help='move computations to gpu')

    return parser.parse_args()

if __name__ == '__main__':

    input_args = get_input_args_train()
    print(input_args)