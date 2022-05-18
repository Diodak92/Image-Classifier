# Imports packages
from os import path
from random import randint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# TODO: Define your transforms for the training, validation, and testing sets

def load_train_valid_data(data_dir):

    train_dir = path.join(data_dir, 'train')
    valid_dir = path.join(data_dir, 'valid')

    # check if path is correct to data folder and train, valid datasets
    if path.exists(data_dir) and path.exists(train_dir) and path.exists(valid_dir):

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

        # TODO: Load the datasets with ImageFolder
        train_dataset = datasets.ImageFolder(train_dir, transform = train_data_transform)
        valid_dataset = datasets.ImageFolder(valid_dir, transform = valid_data_transform)

        # TODO: Using the image datasets and the trainforms, define the dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=False)

        print('Data loaded succesfuly')

        return train_dataloader, valid_dataloader

    else:
        print('Wrong file path!')

if __name__ == '__main__':

    train_data, _ = load_train_valid_data('flower_data')
    images, labels  = next(iter(train_data))
    print(type(images))
    print(images.shape)
    print(labels.shape)