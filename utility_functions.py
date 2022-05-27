# Imports packages
from os import getcwd, path
from random import randint
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
# import own functions and variables
from nn_functions import select_nn_model_arch, optimizer, select_device 

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

    print('Data loaded succesfuly from {}'.format(path.join(getcwd(), data_dir)))

    return train_dataloader, valid_dataloader, class_to_index

# Function for savin the chceckpoint
def save_checkpoint(model_arch,
                    model,
                    optimizer,
                    class_to_index,
                    model_performance,
                    filename='checkpoint'):

    chceckpoint = {'model performance': model_performance,
                   'model architecture' : model_arch,
                   'model state dict': model.state_dict(),
                   'optimizer state': optimizer.state_dict(),
                   'classes to indices': class_to_index
                   }

    # save model state
    filename = path.join('{}_e{}.pth'.format(filename, model_performance['epoches']))
    torch.save(chceckpoint, filename)
    print('Model saved successfully in {}'.format(filename))


# load a chceckpoint
def load_checkpoint(filepath, print_state = False):
    
    '''Load: model performance, model data and optimizer state from file'''
    
    if torch.cuda.is_available():
        state_dict = torch.load(filepath)
    else:
        state_dict = torch.load(filepath, map_location='cpu')
    
    # create nn model and load parameters
    nn_model = select_nn_model_arch(state_dict['model architecture']['model'],
                                    state_dict['model architecture']['hidden units'],
                                    is_pretrained = False)
    nn_model.load_state_dict(state_dict['model state dict'],  strict=False)
    nn_model.eval()
    # load optimizer
    optim = optimizer(nn_model)
    optim.load_state_dict(state_dict['optimizer state'])
    # load class to indexes
    class_to_idx = state_dict['classes to indices']
    # load model performance
    model_performance = state_dict['model performance']
    
    # print state dict
    if print_state:
        for i in state_dict.items():
            print(i, '\n')
    
    print('Checkpoint {} has been successfully loaded!'.format(filepath))
    return nn_model, optim, class_to_idx, model_performance

# load and process image
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    # Process a PIL image for use in a PyTorch model
    with Image.open(image) as im:
        im_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        
        im_tensor = torch.transpose(im_transform(im), 1, 1)
        return np.array(im_tensor)

# function for predicting image top classes and probabilities
def predict(image_path, model, topk=5, gpu = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # import and convert image to tensor
    image = torch.tensor(process_image(image_path))
    # adjust tensor dimentions
    image = torch.unsqueeze(image, 0)
    # move tensor to cpu or gpu
    device = select_device(gpu)
    image = image.to(device)
    with torch.no_grad():
        # set model in evaluation mode
        model.eval()
        # get model probabilities
        prob = torch.exp(model(image))
    # compute and return top probabilities and classes
    top_p, top_class = prob.topk(topk, dim=1)
    
    print('Using {} device\n'.format(device))

    return tuple(np.array(top_p).tolist()[0]), tuple(np.array(top_class).tolist()[0])

# open and load json file that maps the class values to other category  
def map_indexes(file_path):
    with open(file_path, 'r') as f:
        cat_to_name = json.load(f)
        return cat_to_name

if __name__ == '__main__':

    train_data, _ = load_train_valid_data('flower_data')
    images, labels = next(iter(train_data))
    print(type(images))
    print(images.shape)
    print(labels.shape)
