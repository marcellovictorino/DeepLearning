#!/usr/bin/env python
# coding: utf-8

# [ArgParse Documentation](https://docs.python.org/3/library/argparse.html)

# Import required libraries
import argparse
import train_utils


describe_train = """Solution to implement Transfer Learning technique, allowing the user to
specify different neural network architectures, hyperparameters, and others
"""
# Instantiate Parser to listen for CLI inputs
ap = argparse.ArgumentParser(description=describe_train)

# Add Arguments for special functionalities
# Required argument
ap.add_argument('data_dir', type=str, help='folder with proper data training structure') # No default. Required argument!

# Optionl arguments
ap.add_argument('--arch', type=str, help='specify NN architecture: vgg16, resnet50', default='resnet50')
ap.add_argument('--gpu', help='either GPU or CPU usage', default='GPU')
ap.add_argument('--learning_rate', type=float, help='set learning rate', default=0.001)
ap.add_argument('--hidden_units', type=int, help='number of hidden layers before first activation function', default=2048)
ap.add_argument('--epochs', type=int, help='number of complete dataset iterations', default=5)
ap.add_argument('--dropout', type=float, help='layer dropout probability during training', default=0.5)
ap.add_argument('--save_dir', help='directory to save checkpoint file once training is complete', default='Checkpoints')

# Read arguments from command line
arg = ap.parse_args()

data_dir = arg.data_dir
save_dir = arg.save_dir
arch = arg.arch
lr = arg.learning_rate
hidden_units = arg.hidden_units
gpu = arg.gpu
epochs = arg.epochs
dropout = arg.dropout


# 1) load data
print('Loading data...')
train_loader, valid_loader, test_loader, class_to_idx = train_utils.load_data(data_dir, arch)

# 2) Activate GPU if available
device = train_utils.activate_gpu(gpu)

# 3) Set Artificial Neural Network to be trained based on user input
model, optimizer, criterion = train_utils.set_nn(class_to_idx, arch, hidden_units, dropout, lr)

# 4) Actually train the ANN and save summary results to log file
print('Details:')
print(f'    Model: {model.name} | Epochs: {epochs} | Learning Rate: {lr} | Hidden Units: {hidden_units} | *Dropout: {dropout}')
print('Training...')
model = train_utils.train_nn(model, optimizer, criterion, train_loader, valid_loader, test_loader, device, epochs)

# 5) Save best model checkpoint for future loading
train_utils.save_checkpoint(model, optimizer, save_dir)
print('Model saved. Training Completed!')