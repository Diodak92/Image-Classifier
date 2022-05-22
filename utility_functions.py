# Imports packages
import argparse
import numpy as np
from os import path
from random import randint
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
# import own functions and variables
from nn_functions import select_nn_model_arch, optimizer 

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

# Function for data loading and transforms
def load_train_valid_data(data_dir):

    train_dir = path.join(data_dir, 'train')
    valid_dir = path.join(data_dir, 'valid')

    train_data_transform = transforms.Compose([
        transforms.RandomRotation(randint(0, 30)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(
        train_dir, transform=train_data_transform)
    valid_dataset = datasets.ImageFolder(
        valid_dir, transform=valid_data_transform)
    # get clas to indexes
    class_to_index = train_dataset.class_to_idx

    # Using the image datasets and the trainforms, define the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

    print('Data loaded succesfuly')

    return train_dataloader, valid_dataloader, class_to_index

# Function for savin the chceckpoint
def save_checkpoint(model_arch,
                    model,
                    optimizer,
                    class_to_index,
                    model_performance,
                    filepath='checkpoint.pth'):

    chceckpoint = {'model_performance': model_performance,
                   'model architecture' : model_arch,
                   'model state dict': model.state_dict(),
                   'optimizer state': optimizer.state_dict(),
                   'classes to indices': class_to_index
                   }

    # save model state
    torch.save(chceckpoint, filepath)
    print('Model saved successfully!')


# load a chceckpoint
def load_checkpoint(filepath, print_state = False):
    
    '''Load: model performance, model data and optimizer state from file'''
    
    if torch.cuda.is_available():
        state_dict = torch.load(filepath)
    else:
        state_dict = torch.load(filepath, map_location='cpu')
    
    # load chceckpoint data
    model_performance = {}
    model_performance['epoches'] = state_dict['epoches']
    model_performance['train losses'] = state_dict['train losses']
    model_performance['valid losses'] = state_dict['valid losses']

    # create nn model and load parameters
    nn_model = select_nn_model_arch()
    nn_model.load_state_dict(state_dict['model state dict'],  strict=False)
    nn_model.eval()
    # load optimizer
    optim = optimizer(nn_model)
    optim.load_state_dict(state_dict['optimizer state'])
    # load class to indexes
    class_to_idx = state_dict['classes to indices']
    
    # print state dict
    if print_state:
        for i in state_dict.items():
            print(i, '\n')
    
    print('Data has been successfully loaded')
    return nn_model, optim, class_to_idx, model_performance

# load and process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # TODO: Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        im_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        im_tensor = torch.transpose(im_transform(im), 1, 1)
        return np.array(im_tensor)

# function for predicting image top classes and probabilities
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # import and convert image to tensor
    image = torch.tensor(process_image(image_path))
    # adjust tensor dimentions
    image = torch.unsqueeze(image, 0)
    # move tensor to cpu or gpu
    image = image.to(device = 'cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        # set model in evaluation mode
        model.eval()
        # get model probabilities
        prob = torch.exp(model(image))
    # compute and return top probabilities and classes
    top_p, top_class = prob.topk(topk, dim=1)
    
    return tuple(np.array(top_p).tolist()[0]), tuple(np.array(top_class).tolist()[0])


if __name__ == '__main__':

    train_data, _ = load_train_valid_data('flower_data')
    images, labels = next(iter(train_data))
    print(type(images))
    print(images.shape)
    print(labels.shape)

    input_args = get_input_args_train()
    print(input_args)
