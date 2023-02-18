# Imports packages
from torch import nn
from menu import get_input_args_train
from utility_functions import load_train_valid_data, save_checkpoint
from nn_functions import select_nn_model_arch, optimizer, select_device, train_and_valid_nn

# get user input arguments from command line
input_args = get_input_args_train()

# check if path is correct to data folder
try:
    # get training and validation dataloader
    train_data, valid_data, class_to_index, n_classes = load_train_valid_data(input_args.dir)
    # select model architecture optimizer and computation device for training
    nn_model = select_nn_model_arch(archName = input_args.arch,
                                    hiddenUnits = input_args.hidden_units,
                                    classesNumber = n_classes)
    optim = optimizer(nn_model, input_args.learning_rate)
    device = select_device(input_args.gpu)

    # store model architecture and params
    model_arch = {'model': input_args.arch, 'hidden units': input_args.hidden_units}
    # data container for storing model performance while training
    train_performance = {'epoches': input_args.epochs, 'train losses': [], 'valid losses' : []}

    # train and valid neural network classifier
    train_and_valid_nn(train_data,
                    valid_data,
                    nn_model,
                    nn.NLLLoss(),
                    optim,
                    device,
                    train_performance)

    # save trained model to a file 
    save_checkpoint(model_arch,
                    nn_model,
                    optim,
                    class_to_index,
                    train_performance,
                    input_args.save_dir)

except (NameError, FileNotFoundError):
    print('Wrong file path!')
