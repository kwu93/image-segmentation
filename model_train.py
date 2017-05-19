import tensorflow as tf
import numpy as np
from data_pipeline import create_data_pipeline
from util import *
from convnet import ConvNet
from image_segmentation_model import ImageSegmentation


""" 
Main function that holds model hyperparameters (probably more suited to be populated from tf.app.flags).
Performs initialization of models and retrieving data from data pipeline.
Calls the train() function of the ImageSegmentation model
"""

class Config:
    '''Holds model hyperparameters and data information.'''
    contour_width = 256 # size of the boolean mask 
    contour_height = 256
    image_width = 256 # size of dicom image 
    image_height = 256
    output_size = 256 * 256
    filter_size = 5
    batch_size = 8
    epochs = 10
    max_gradient_norm = 5 
    optimizer = 'adam'
    learning_rate = 1e-4
    clip_gradients = False
    dropout = 0.0

    def __init__(self):
        pass   


def main(_):
    config = Config()

    print "Initializing Convolutional Neural Network...\n"
    cnn = ConvNet(256,256)

    print "Initializing Image Segmentation Model...\n"
    model = ImageSegmentation(config, cnn)

    print "Retrieving data from pipeline...\n"
    data = create_data_pipeline()


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print "Beginning to train model\n"
        train_loss = model.train(sess, data, 'train_dir', 2)

    print "Finished"

if __name__ == "__main__":
    tf.app.run()
