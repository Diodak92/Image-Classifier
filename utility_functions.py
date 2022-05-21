# Imports packages
import argparse
from os import path
from random import randint
from torch import save
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Create command line argument parser
def get_input_args():
    parser = argparse.ArgumentParser(
        description='Train a neural network on a data set and save the model to a checkpoint file')
    # get directory of images folder
    parser.add_argument('dir', type=str, default='flowers',
                        help='path to the folder of images dataset')
    # get directory to save checkpoint file
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth',
                        help='set directory to save checkpoint')
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
def save_checkpoint(model,
                    optimizer,
                    class_to_index,
                    model_performance,
                    filepath='checkpoint.pth'):

    chceckpoint = {'epoches': model_performance['epoches'],
                   'train losses': model_performance['train losses'],
                   'valid losses': model_performance['valid losses'],
                   'model state dict': model.state_dict(),
                   'optimizer state': optimizer.state_dict(),
                   'classes to indices': class_to_index
                   }

    # save model state
    save(chceckpoint, filepath)
    print('Model saved successfully!')


if __name__ == '__main__':

    train_data, _ = load_train_valid_data('flower_data')
    images, labels = next(iter(train_data))
    print(type(images))
    print(images.shape)
    print(labels.shape)
