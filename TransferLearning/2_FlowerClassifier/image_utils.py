#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports here
import torch
# from torch import nn
# from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from time import time
from PIL import Image
import numpy as np
# import copy
import seaborn as sns

import train_utils
import json

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


def load_checkpoint(checkpoint):
    '''
    This function loads saved parameters and state from previously trained Neural Network.
    Input: file name of checkpoint
    Output: model, optimizer, device (GPU or CPU), loss criterion, and checkpoint dictionary with secondary information
    '''
    # Read from checkpoint.pth saved file
    filename = checkpoint #f'checkpoint2_{model_name}.pth'
    
    # Deserialize 'pickled' file (reading saved checkpoint)
    ### Tip: use map location to enable run on CPU model trained in GPU
    checkpoint = torch.load(filename, map_location=lambda storage, loc: storage)
    
    # Initialize model, applying custom setup for each one
    model, optimizer, criterion = train_utils.set_nn(checkpoint['class_to_idx'], checkpoint['model_name'])
    
    # load model and optimizer saved state data
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer, criterion, checkpoint


# In[15]:


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array, as expected by Torch.Tensor.
    '''
    # resize to 256 pixel
    image = Image.open(image_path).resize((256,256))

    # Center crop to 224 pixel
    width, height = image.size   # Get dimensions
    final_size = 224

    left = (width - final_size)/2
    top = (height - final_size)/2
    right = (width + final_size)/2
    bottom = (height + final_size)/2

    image = image.crop((left, top, right, bottom))

    # Transform image into np.array
    im = np.array(image)

    # Normalize pixels from [0 - 255] to [0 - 1.0] float range
    im = (im - im.min()) / (im.max() - im.min())

    # Normalize as expected by pre-trained model
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    im = (im - mean ) / std
    
    # Transpose moving color channel from third (matplotlib) to first position (pytorch)
    im = im.transpose((2, 0, 1)) # color, x, y
    
    return im


# In[16]:


def imshow(image, ax=None):
    """Transforms back from Tensor to Image format and display."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
        
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (std * image) + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    # Remove spines/axis
    ax.axis('off')
    
    ax.imshow(image)
    
    return ax


# In[ ]:


def load_cat_to_name(cat_to_name_file='cat_to_name.json'):

    with open(cat_to_name_file, 'r') as f:
        cat_to_name = json.load(f)
        
    return cat_to_name


# In[143]:


def predict(image_path, model, device, cat_to_name, k=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Bring model to device (CPU or GPU, if available)
    model.to(device)
    # Make sure model is ready to be used for classifying/evaluation
    model.eval()
    
    # Transform original image into numpy array, as expected by Torch.Tensor
    processed_image = process_image(image_path)

    # Load processed image as Tensor. Fix: cast input image as float
    input_tensor = torch.tensor(processed_image).float()
        
    # Use GPU if available
    input_tensor = input_tensor.to(device)   
    
#     # As recommended, convert input image to FloatTensor
#     input_tensor = input_tensor.float()
    
    # Add expected batch information for a single image
    input_tensor = input_tensor.unsqueeze_(0)

    output = model.forward(input_tensor)

    probabilities = torch.exp(output)
    top_p, top_class = probabilities.topk(k, dim=1)

    # unpack from Tensor back to simple list
    top_class = top_class.squeeze().tolist()
    top_p = top_p.squeeze().tolist()
        
    # Convert indices to actual classes
    idx_to_class = {val: key for key,val in class_to_idx.items()}
    
    top_label = [idx_to_class[class_] for class_ in top_class]
    top_flower = [cat_to_name[label] for label in top_label]
        
    return top_p, top_label, top_flower


# In[145]:


# Display an image along with the top 5 classes
import seaborn as sns
def display_result(image_path, model):
    
    fig, axes = plt.subplots(2,1, figsize=(5,8))
    
    # Set up title
    flower_num = image_path.split('/')[2]
    title = checkpoint['class_to_idx'].get(str(flower_num))
        
    # Plot flower
    img = process_image(image_path)
    axes[0].set_title(title)
    imshow(img, ax=axes[0]);
    
    # Make prediction
    probs, classes, flowers = predict(image_path, model)
    
    # Plot bar chart
    sns.barplot(x=probs, y=flowers, ax=axes[1],color=sns.color_palette()[0])
    plt.show();

