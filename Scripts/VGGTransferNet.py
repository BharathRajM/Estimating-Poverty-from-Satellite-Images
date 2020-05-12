#!/usr/bin/env python
# coding: utf-8
# -*- coding: utf-8 -*-
"""
Created on Thu May 6 03:51:17 2020
@author: Bharath
Code for all API calls made from python.
"""


# In[1]:


import keras
from keras.applications.vgg16 import VGG16
from keras import Input
from keras import models
from keras import optimizers
from keras import layers
from keras import backend as K

# In[2]:


class VGGTransferNet:
    @staticmethod
    
    def build(width=400, height=400, depth=3, classes=3):
        
        #create a sequential model
        model = models.Sequential()
        
        #define the input dimensions
        input_format = Input(shape=(400,400,3),name = 'image_input')        
        
        #use the pretrained model until the last conv-pool block
        vgg_model = VGG16(weights = 'imagenet',include_top=False, input_tensor=input_format)
        
        #freeze the layers for training
        for layer in vgg_model.layers:
            layer.trainable = False
        
        #add the VGG convolutional base model
        model.add(vgg_model)
        
        #add new 3 layers of convolution and then avg_pooling and then softmax
        #filters = 4096 kernel size = (6,6), stride = 6
        model.add(layers.Conv2D(filters=4096, kernel_size=(6,6), activation='relu',strides = (6,6))) 

        #filters = 4096 kernel size = (1,1), stride = 1
        model.add(layers.Conv2D(filters=4096, kernel_size=(1,1), activation='relu',strides = (1,1)))

        #filters = 3 kernel size = (1,1), stride = 1
        model.add(layers.Conv2D(filters=3, kernel_size=(1,1), activation='relu',strides = (1,1)))

        #add avg_pooling layer pool_size=(2,2)
        model.add(layers.AvgPool2D(pool_size=(2,2)))
        
        #flatten the output
        model.add(layers.Flatten())
        
        #add softmax
        model.add(layers.Dense(units=3, activation="softmax"))
        
        return model


# In[ ]:




