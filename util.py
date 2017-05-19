"""
Util file with helper functions
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dicom 
import pylab
from parsing import *


"""
Returns the TensorFlow optimizer operation based off the name 
:param data: tuple of (numpy array for images, numpy array for targets). numpy arrays are [num_data_points, pixel_width, pixel_height]
:param minibatch_size: the batch_size
:param shuffle: randomize data or not when placing into batches. Default is true.
:return: list of batched data
"""
def get_minibatches(data, minibatch_size, shuffle=True):
    num_samples = len(data[0])
    if(shuffle):
        indices = np.random.permutation(num_samples) 
    else:
        indices = np.arange(num_samples)

    minibatches = []
    for minibatch_start in np.arange(0, num_samples, minibatch_size):
        minibatch_end = min(num_samples, minibatch_start + minibatch_size)
        minibatch_indices = indices[minibatch_start:minibatch_end]

        dicom_images = data[0][minibatch_indices]
        boolean_targets = data[1][minibatch_indices]
        
        minibatch = [dicom_images, boolean_targets]
        minibatches.append(minibatch)

    return minibatches


"""
Returns the TensorFlow optimizer operation based off the name 
:param opt: name of opt e.g. adam or sgd
:return: None
"""
def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


"""
Plots a boolean mask, which represents the contours 
:param mask: - boolean array of True / False values 
:return: None
"""
# http://stackoverflow.com/questions/9638826/plot-a-black-and-white-binary-map-in-matplotlib
def display_boolean_mask(mask):
    plt.imshow(mask)
    plt.imshow(mask, cmap='Greys', interpolation = 'none')
    plt.show()

"""
Displays the image contents of a DICOM file 
:param dcm_file: takes in name of file AND the patient_id e.g. 'SCD0000101/1.dcm'
:return: None
"""
def visualize_dicom(dcm_file): 
    dcm_path = DICOM_DIR + dcm_file
    ds=dicom.read_file(dcm_path)
    pylab.imshow(ds.pixel_array, cmap=pylab.cm.bone)
    pylab.show()
    