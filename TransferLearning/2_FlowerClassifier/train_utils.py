#!/usr/bin/env python
# coding: utf-8

# This is the supporting script with the main steps used while training an Artificial Neural Network using the Transfer Learning approach on the pre-trained models made available by PyTorch.

# Imports here
import torch
from torch import nn
from torchvision import datasets, transforms, models

from time import time
import copy
import os


def load_data(data_dir='flowers', arch='vgg16'):
    """
    data_dir: directory containing all images, following the classical train/valid/test subfolder structure recommend by PyTorch\n
    arch: specify the architecture of the pre-trained model that will be used for Transfer Learning. Defaults to 'vgg16'\n
    > Obs.: The Inception architecture requires the training image to have at least 299 pixel instead of the typical 224.
    """
    # Define train/validation/test folder structure
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Normalize the means and std for all images to match Pre-trained network
    
    # Note: Inception architecture requires at least 299 pixels for final image instead of typical 224
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                          transforms.RandomResizedCrop(299 if arch.lower().startswith('inception') else 224), 
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

    test_transforms = transforms.Compose([transforms.Resize(320 if arch.lower().startswith('inception') else 255),
                                          transforms.CenterCrop(299 if arch.lower().startswith('inception') else 224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 

    # Load the datasets with ImageFolder and apply transforms
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Define the dataloaders for train, validation, and test
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Save the ordered index sequence as interpreted by the model while training. Required to later convert back to the correct Class/Label
    class_to_idx = train_data.class_to_idx
    
    return train_loader, valid_loader, test_loader, class_to_idx


# Activate GPU use if available and requested
def activate_gpu(gpu='GPU'):
    """Use GPU if available and requested by user. Defaults to use GPU if available."""
    if torch.cuda.is_available() and gpu.lower() == 'gpu':
        print('Running on GPU')
        device = torch.device('cuda:0')
    else:
        print('Running on CPU')
        device = torch.device('cpu')
        
    return device


def load_pre_trained(class_to_idx, arch='vgg16'):
    """Download and returns one of the available pre-trained models.\n
    + arch = architecture to be loaded based on user input\n
    + class_to_idx: dictionary resulting from function load_data()\n\n
    
    Return pre-trained model.
    """
    # NOTE: for some reason, using dictionary did not work. Kept going through all elements.
    if arch.lower() == 'vgg16':
        model = models.vgg16(pretrained=True)
    
    if arch.lower() == 'vgg19':
        model = models.vgg19(pretrained=True)
        
    if arch.lower() == 'resnet50':
        model = models.resnet50(pretrained=True)
        
    if arch.lower() == 'inception_v3':
        model = models.inception_v3(pretrained=True)
    
    # Download and Load pre-trained model
    try:
        model.class_to_idx = class_to_idx
        return model
    except:
        print(f'Selected architecture not recognized. Please, select one of: vgg16, vgg19, resnet50, or inception_v3')


def set_nn(class_to_idx, arch='vgg16', hidden_units=2048, dropout=0.5, lr=0.001):
    """Set up the Artificial Neural Network structure to implement Transfer Learning technique:\n
    1) Load pre-trained model,\n
    2) Freeze all but the last layer for training on images of interest,\n
    3) Define parameters of last layer based on specific architecture, using the number of hidden units based on user input
    4) Define the Adam optimizer to be used for training the ANN, using the Learning Rate entered by user. Defaults to 0.001\n\n
    
    Returns the final model structure and optimizer based on user input.
    """ 
    # Load pre-trained model
    model = load_pre_trained(class_to_idx, arch)
    
    # Freeze parameters to not backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Redefine last layer to train on images/classes of interest
    if arch.lower().startswith('vgg'):
        # Save the number of input faetures for future Loading from Checkpoint
        model.input_features = model.classifier[0].in_features
        
        model.classifier = nn.Sequential(nn.Linear(model.input_features, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(dropout), # probability to randomly drop out a hidden layer
                                 nn.Linear(hidden_units, len(model.class_to_idx)),
                                 nn.LogSoftmax(dim=1))
        
        # Only train the classifier parameters
        optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)    
        
    elif arch.lower().startswith('resnet'):
        model.input_features = model.fc.in_features
        
        model.fc = nn.Sequential(nn.Linear(model.input_features, len(model.class_to_idx)),
                                 nn.LogSoftmax(dim=1))
        # Only train the classifier parameters
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
        
    elif arch.lower().startswith('inception'):
        model.input_features = model.fc.in_features
        
        model.fc = nn.Sequential(nn.Linear(model.input_features, len(model.class_to_idx)),
                                 nn.LogSoftmax(dim=1))
        
        # Requirements to use Inception_v3
        model.aux_logits = False
        
        # Only train the classifier parameters
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    
    # Define criteria to evaluate model loss
    criterion = nn.NLLLoss() # Negative Log Likelihood Loss
    
    model.name = arch.lower() # Saves the architecture used for future reference
        
    return model, optimizer, criterion


def train_nn(model, optimizer, criterion, train_loader, valid_loader, test_loader, device, epochs=5):
    """Trains the Artificial Neural Network, printing the training summary for each epoch and final results.\n
    A training log is saved as text file.\n
    + model: predefined model to trained\n
    + optimizer: predefined Adam optimizer\n
    + criterion: predefined loss criteria\n
    + train_loader: predefined loader with images to be used for training\n
    + valid_loader: predefined loader with images to be used for validation\n
    + test_loader: predefined loader with images to be used for testing\n
    + device: predefined state to use GPU if available
    + epochs: number of epochs to perform training, defined by user input. Defaults to 5.\n\n
    
    Returns the best model based on validation accuracy.
    """
    best_model = 0
    best_score = 0

    # Define Learning Rate Decay using scheduler - as recommended by PyTorch
    ### Every step_size epochs, lr = lr * gamma
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Use GPU if available
    model = model.to(device)
    
    start = time()
    
    with open(f'trainingLog_{model.name}.txt', 'w') as log:
        for epoch in range(epochs):
            # Training Stage
            training_losses = []
            validation_losses = []
            running_loss = 0
            model.train() #!Important: set train mode, applying dropout etc.

            for inputs, labels in train_loader:

                # !Important: restart optimizer for every new batch
                optimizer.zero_grad()

                # move input and label tensors to the appropriate device (GPU or CPU)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Perform forward pass to transform input into output
                outputs = model.forward(inputs) # since last layer is LogSoftMax, output is Log

                # Calculate error. Using Negative Log Loss instead of delta probability
                loss = criterion(outputs, labels)

                # Back propagate
                loss.backward()

                # Calibrate Weights based on Gradient Descent
                optimizer.step()

                # Keep track of training progress
                running_loss += loss.item()
                training_losses.append(running_loss/len(train_loader))

            # Validation Stage
            validation_loss = 0
            accuracy = 0
            model.eval() #! Important: enter evaluation mode, no drop-out applied

            with torch.no_grad(): # ! Important: no gradient descent on following calculations
                for inputs, labels in valid_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    predictions = model.forward(inputs)
                    batch_loss = criterion(predictions, labels)
                    validation_loss += batch_loss.item()

                    validation_losses.append(validation_loss/len(valid_loader))

                    # Calculate Accuracy
                    probabilities = torch.exp(predictions) #proba = exp^(log)
                    top_p, top_class = probabilities.topk(1, dim=1) # returns class with higher probability score

                    equals = (top_class == labels.view(*top_class.shape)) # True (1) or False (0)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    # Evaluate model performance and keep only the best
                    if accuracy/len(valid_loader) > best_score:
                        best_score = accuracy/len(valid_loader)
                        best_model = copy.deepcopy(model)

            current_lr = optimizer.param_groups[0]['lr']

            results_summary = (f'Epoch [{epoch+1}/{epochs}] '
                               f'Learning Rate: {current_lr:.6f} | '
                              f'Train Loss: {running_loss/len(train_loader) :.3f} | '
                              f'Validation Loss: {validation_loss/len(valid_loader) :.3f} | '
                              f'Validation Accuracy: {accuracy/len(valid_loader) :.2%} | '
                              f'Time: {(time()-start):.0f} s\n')
            
            # Print and save to log
            print(results_summary)
            log.write(results_summary)

            # take a step on scheduler for learning rate decay
            scheduler.step()

        # Load best model identified during training/validation process
        model = best_model
        
        best_model_result = f'    --> Best Validation Accuracy: {best_score:.2%}\n'
        print(best_model_result)
        log.write(best_model_result)
        
        ###############################################################################
        # Testing
        
        test_section = '\n########## TESTING ##########\n'
        print(test_section)
        log.write(test_section)
        
        testing_accuracy = []

        model.eval() #! Important: enter evaluation mode, no drop-out applied
        start = time()

        with torch.no_grad(): # ! Important: no gradient descent on following calculations
            for inputs, labels in test_loader: 
                test_loss = 0
                accuracy = 0 

                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model.forward(inputs)
                batch_loss = criterion(outputs, labels)
                test_loss += batch_loss.item()

                # Calculate Accuracy
                probabilities = torch.exp(outputs) #proba = exp^(log)
                top_p, top_class = probabilities.topk(1, dim=1) # returns class with higher probability score

                equals = (top_class == labels.view(*top_class.shape)) # True (1) or False (0)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                testing_accuracy.append(accuracy)
                
                test_result = f'Test Loss: {test_loss:.3f} | Test Accuracy: {accuracy:.2%} | Time: {(time()-start):.0f} sec\n'
                print(test_result)
                log.write(test_result)

        test_summary = f'\n    --> Overall average Test Accuracy: {sum(testing_accuracy)/len(testing_accuracy):.2%}\n'
        print(test_summary)
        log.write(test_summary)  
        
        # Save hyperparameter info for future use
        model.epochs = epochs
    
    return model


def save_checkpoint(model, optimizer, save_dir='Checkpoints'):
    """Save trained model as checkpoint file for later reference: `checkpoint_model_name.pth`"""
    checkpoint = {'epochs': model.epochs,
                  'model_name': model.name,
                  'input': model.input_features,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(checkpoint, os.path.join(save_dir, f'checkpoint_{model.name}.pth'))