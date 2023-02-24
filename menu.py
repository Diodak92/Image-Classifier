# Imports packages
import argparse
import os

# Create command line argument parser for train script
def get_input_args_train():
    parser = argparse.ArgumentParser(prog='train',
        description='Train a neural network on a data set and save the model to a checkpoint file')
    # get directory of images folder
    parser.add_argument('dir', type=str, default=os.path.join(os.getcwd(),'flowers'),
                        help='path to the folder of images dataset (default: %(default)s)')
    # get directory to save checkpoint file
    parser.add_argument('-s', '--save_dir', type=str, default=os.path.join(os.getcwd(), 'checkpoint.pth'),
                        help='set directory to save checkpoint file ex: "cwd/filepath/checkpoint.pth" (default: %(default)s)')
    # get CNN model architecture
    parser.add_argument('--arch', type=str, default='alexnet', choices=['alexnet', 'densenet', 'vgg16'],
                        help="CNN model architecture: densenet, alexnet, or vgg16 (default: %(default)s)")
    # get learning rate
    parser.add_argument('-lr', '--learning_rate', nargs='?', type=float, default=0.001,
                        help='select learning rate: a < 1 (default: %(default)s)')
    # get size of hidden layer for classifier
    parser.add_argument('-hu', '--hidden_units', type=int, default=512,
                        help='set number of hidden units (default: %(default)s)')
    # get size number of epoches
    parser.add_argument('-e', '--epochs', type=int, default=20,
                        help='set number of epoches (default: %(default)s)')
    # select gpu as computation device
    parser.add_argument('--gpu', action='store_true',
                        help='Accelerate the learning by moving computation to the GPU')

    return parser.parse_args()


# Create command line argument parser for predict script
def get_input_args_predict():
    parser = argparse.ArgumentParser(
        description='The program predicts the object in a given image')
    # get directory of image to predict
    parser.add_argument('dir', type=str,
                        help='path to the image')
    # get directory to save checkpoint file
    parser.add_argument('checkpoint', type=str, default='densenet_e20.pth',
                        help='path to load the learned neural network model')
    # get number of most likely classes to return
    parser.add_argument('--top_k', type=int,
                        help='return top K most likely classes with the probabilities')
    # get learning rate
    parser.add_argument('-cn', '--category_names', type=str,
                        help='select JSON file for mapping of categories to real names')
    
    # select gpu as computation device
    parser.add_argument('--gpu', action='store_true',
                        help='Accelerate image recognition by moving computation to the GPU ')

    return parser.parse_args()
