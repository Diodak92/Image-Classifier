import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import models


# define optimizer for neural network classifier and set learning rate
def optimizer(nn_model, learningRate = 0.001):
    return optim.Adam(nn_model.classifier.parameters(), lr = learningRate)

# model training
def train_nn(dataloader, model, loss_fn, optimizer, device):
    
    # store running loss
    running_loss = 0
    
    # set model in training mode
    model.train()
    
    for images, labels in dataloader:
        
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
            accuracy  += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return (test_loss/len(dataloader), accuracy/len(dataloader))

# train neural network
def train_nn_epoches(train_dataloader,
                     valid_dataloader,
                     model, 
                     criterion, 
                     optimizer,
                     device,
                     model_performance):
    
    print('Neural network training has begun!\nBe patient it may take a while...\n')
    print('Using {} device\n'.format(device))
    
    # move model to GPU or CPU
    model.to(device)
    
    epochs = model_performance['epoches']
    
    for epoch in range(epochs):

        train_loss = train_nn(train_dataloader, model, criterion, optimizer, device)
        valid_data = test_nn(valid_dataloader, model, criterion, device)

        (valid_loss, accuracy) = valid_data

        # store testing performance
        model_performance['train losses'].append(train_loss)
        model_performance['valid losses'].append(valid_loss)

        print('Epoch: {}/{};'.format(epoch+1, epochs),
              'Train loss: {:.3f};'.format(train_loss),
              'Validation loss: {:.3f};'.format(valid_loss),
              'Validation accuracy: {:.2f}%'.format(accuracy*100.0))

    print('Training done!') 