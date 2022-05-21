# Imports packages
from click import option
from torch import nn
from utility_functions import get_input_args, load_train_valid_data, save_checkpoint
from nn_functions import select_nn_model_arch, optimizer, select_device, train_and_valid_nn

input_args = get_input_args()
print(input_args)
#print(input_args.gpu)

train_data, valid_data = load_train_valid_data('flower_data')

nn_model = select_nn_model_arch('alexnet')
optim = optimizer(nn_model, learningRate = 0.001)
device = select_device('gpu')

train_performance = {'epoches': 20, 'train losses': [], 'valid losses' : []}

train_and_valid_nn(train_data,
                   valid_data,
                   nn_model,
                   nn.NLLLoss(),
                   optim,
                   device,
                   train_performance)

save_checkpoint(nn_model,
                optim,
                train_data,
                train_performance,
                'alexnet_checkpoint.pth')
