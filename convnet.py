import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf 
import matplotlib.pyplot as plt
from parsing import *
from os import listdir
from os.path import isfile, join
import os

"""
Very basic convolutional neural network which includes convolutional layers with RELU activation and 
optionally pooling layers.  

Current architecture almost certainly not useful for the image segmentation task. However, serves as
a template model. 

Comment: ImageSegmentation task might be served well with deconvolutional and unpooling layers. 
Reference: <http://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf>

This model is derived from 
<https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py>
with changed parameters, formatting, and object-oriented flavor. 
"""
class ConvNet:
# Conv-net architecture inspired by
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/convolutional_network.py
# 
    def __init__(self, output_width, output_height, dropout = 0.0, namespace = "convnet"):
        # Store layers weight & bias
        with tf.variable_scope("cnn", reuse = None):
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.output_width = output_width
            self.output_height = output_height
            self.output_size = output_width * output_height
            
            self.weights = {
            'wc1': tf.get_variable(name='wc1',dtype = tf.float32, shape = [5,5,1,4], initializer = xavier_init),
            'wc2': tf.get_variable(name='wc2',dtype = tf.float32, shape = [5,5,4,8], initializer = xavier_init),
            'wd1': tf.get_variable(name='wd1',dtype = tf.float32, shape = [64*64*8,1024], initializer = xavier_init),
            'out': tf.get_variable(name='wout',dtype = tf.float32, shape = [1024, self.output_size], \
                                    initializer = xavier_init)
            }

            self.biases = {
            'bc1': tf.get_variable(name='bc1', dtype = tf.float32, shape = [4], initializer = tf.constant_initializer(0)),
            'bc2': tf.get_variable(name='bc2', dtype = tf.float32, shape = [8], initializer = tf.constant_initializer(0)),
            'bd1': tf.get_variable(name='bd1', dtype = tf.float32, shape = [1024], initializer = tf.constant_initializer(0)),
            'out': tf.get_variable(name='bout', dtype = tf.float32, shape = [self.output_size], \
                                    initializer = tf.constant_initializer(0))
            }
    
    def conv2d(self, x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)


    def maxpool2d(self, x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')

    def forwardProp(self,X, dropout = 0.0):
        # Reshape input picture
        tf.expand_dims(X, -1) # x goes from [batch_size, width, height] -> [batch, width, height, 1] i.e. depth = 1
        # Convolution Layer
        weights = self.weights
        biases = self.biases
        
        print "got here just ifne"
        conv1 = self.conv2d(X, weights['wc1'], biases['bc1'])
        # OPTIONAL: Max Pooling (down-sampling) -- unused here
        conv1 = self.maxpool2d(conv1, k=2)

        # Convolution Layer
        conv2 = self.conv2d(conv1, weights['wc2'], biases['bc2'])
        # OPTIONAL: Max Pooling (down-sampling) -- unused here
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        #fc1 = tf.nn.dropout(fc1, dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        
        out = tf.reshape(out, [-1, self.output_width, self.output_height])
        return out