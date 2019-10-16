#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Import required libraries
import argparse
import train_utils


# In[7]:


describe_predict = """Solution to implement Transfer Learning technique, allowing the user to
load previously trained models to classify images
"""
# Instantiate Parser to listen for CLI inputs
ap = argparse.ArgumentParser(description=describe_predict)

# Add Arguments for special functionalities
# Required argument
ap.add_argument('image_file', type=str, help='path to image to be classified')
ap.add_argument('checkpoint', type=str, help='previously trained model to be loaded')

# Optionl arguments
ap.add_argument('--top_k', type=int, help='returns the predicted label and its probability of the Top K classes', default=5)
ap.add_argument('--category_names', type=str, help='JSON file mapping the categories to real label names', default='cat_to_name.json')
ap.add_argument('--gpu', type=str, help='either GPU or CPU for inference', default='GPU')

# Read arguments from command line
arg = ap.parse_args()

image_file = ap.image_file
checkpoint_file = ap.checkpoint_file
gpu = ap.gpu
topk = ap.top_k
cat_to_name = ap.category_names


# In[ ]:


# 1) load Checkpoint
print('Loading checkpoint...')
model, optimizer, criterion, checkpoint = image_utils.load_checkpoint(checkpoint_file)


# In[ ]:


# 2) Activate GPU if available
device = train_utils.activate_gpu(gpu)


# In[ ]:


# 3) Load mapping from category to label name
cat_to_name = image_utils.load_cat_to_name(cat_to_name)


# In[ ]:


# 4) Run image through model and make prediction
print(f'Classifying image: {image_file}\n')
top_p, top_cat, top_name = image_utils.predict(image_file, model, device, cat_to_name, topk)

for i in enumerate(zip(top_p, top_cat, top_name)):
    print(f'Top {i[0]}) Name: {i[1][2]} | Class: {i[1][1]} | Probability: {i[1][0]:.2%}')

