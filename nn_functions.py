import torch
from torch import nn, optim
from torchvision import models as M
from tqdm import tqdm


# return selected model and freeze features
def select_nn_model_arch(archName, hiddenUnits = 512, classesNumber = 102):
    
    # sample neural network models
    densenet = M.densenet161(weights = M.DenseNet161_Weights.DEFAULT)
    alexnet = M.alexnet(weights = M.AlexNet_Weights.DEFAULT)
    vgg16 = M.vgg16(weights = M.VGG16_Weights.DEFAULT)
    
    # models dict
    my_models = {'densenet': {'model' : densenet, 'in_features' : 2208},
                 'alexnet': {'model' : alexnet, 'in_features' : 9216},
                 'vgg16': {'model' : vgg16, 'in_features' : 25088}}
    
    # create and process the selected model 
    model = my_models[archName]['model']
    # freeze model features
    for param in model.parameters():
        param.requires_grad = False
    
    # create model classifier
    model.classifier = nn.Sequential(nn.Linear(my_models[archName]['in_features'], hiddenUnits),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(p=0.3, inplace=False),
                                        nn.Linear(hiddenUnits, classesNumber),
                                        nn.LogSoftmax(dim=1))
    return model

# define optimizer for neural network classifier and set learning rate
def optimizer(nn_model, learningRate=0.001):
    return optim.Adam(nn_model.classifier.parameters(), lr=learningRate)

# function for selecting computing device
def select_device(device = False):
    if device:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        return 'cpu'

# model training
def train_nn(dataloader, model, loss_fn, optimizer, device, training_progress):

    # store running loss
    running_loss = 0

    # set model in training mode
    model.train()

    for _, data in enumerate(tqdm(dataloader, desc = 'Epoch: {}/{}'.\
                                  format(training_progress['epoch']+1, training_progress['epoches']))):

        images, labels = data
        # move computations
        images, labels = images.to(device), labels.to(device)

        # forward pass through the network
        pred = model(images)
        # calculate the loss
        loss = loss_fn(pred, labels)

        # clear the gradient
        optimizer.zero_grad()
        # perform a backward pass
        loss.backward()
        # update the weights
        optimizer.step()

        # extract and accumulate loss value
        running_loss += loss.item()

    return running_loss/len(dataloader)

# model testing
def test_nn(dataloader, model, loss_fn, device):

    # initialize variable to track model performance
    test_loss, accuracy = 0, 0
    # set model to evaluation mode
    model.eval()

    # turn off gradient
    with torch.no_grad():
        for images, labels in dataloader:
            # move computations
            images, labels = images.to(device), labels.to(device)
            # forward pass through the network
            pred = model(images)

            # extract and accumulate loss value
            test_loss += loss_fn(pred, labels).item()

            # get the class probabilities
            ps = torch.exp(model(images))
            # return top probabilietes and classes
            _, top_class = ps.topk(1, dim=1)
            # check number of matches for  pred / labels
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return (test_loss/len(dataloader), accuracy/len(dataloader))

# train neural network
def train_and_valid_nn(train_dataloader,
                       valid_dataloader,
                       model,
                       criterion,
                       optimizer,
                       device,
                       model_performance):

    print('Neural network training has started!\nBe patient it may take a while...\n')
    print('Using {} device\n'.format(device))

    # move model to GPU or CPU
    model.to(device)

    epochs = model_performance['epoches']

    for epoch in range(epochs):

        training_progress = {"epoch" : epoch, 'epoches' : epochs}

        train_loss = train_nn(train_dataloader, model,
                              criterion, optimizer, device, training_progress)
        valid_data = test_nn(valid_dataloader, model, criterion, device)

        (valid_loss, accuracy) = valid_data

        # store testing performance
        model_performance['train losses'].append(train_loss)
        model_performance['valid losses'].append(valid_loss)

        print('Train loss: {:.3f};'.format(train_loss),
              'Validation loss: {:.3f};'.format(valid_loss),
              'Validation accuracy: {:.2f}%'.format(accuracy*100.0))

    print('Training done!')

if __name__ == '__main__':

    nn_model = select_nn_model_arch('alexnet')
    print(nn_model.classifier)
    device = select_device('GPU')
    print(device)