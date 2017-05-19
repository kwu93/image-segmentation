"""
Constructs data pipeline, parsing through DICOM files and contours and matching them together 
to be passed into our deep learning model. 

Main function is at the end: create_data_pipeline() - all others are helpers
"""
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf 
import matplotlib.pyplot as plt
from parsing import *
from os import listdir
from os.path import isfile, join
import os

DATA_DIR = 'final_data/'
DICOM_DIR = DATA_DIR + 'dicoms/'
CONTOUR_DIR = DATA_DIR + 'contourfiles/'
LINK_FILE = DATA_DIR + 'link.csv'

def process_link_file(fileName):
    f = open(fileName, 'rb')
    header = f.readline()
    links = {}
    for line in f:
        patient_id, original_id =  line.strip('\n').split(',')
        links[patient_id] = original_id
    return links

def process_contour_file(filename, flag = 'inner', width = 256, height =256):
    CONTOUR_TYPE = 'i-contours' if (flag == 'inner') else 'o-contours'
    if '/' not in filename:
        filepath = CONTOUR_DIR + filename + "/" + CONTOUR_TYPE + "/"
    else:
        filepath = filename
    contour_file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]

    coords_list = []
    for contour_file in contour_file_list:
        contour_filepath = filepath + contour_file
        coords_list.extend(parse_contour_file(contour_filepath))
    
    boolean_mask = poly_to_mask(coords_list, width, height)
    return boolean_mask


def process_dicom_file(filename):
    if '/' not in filename:
        filepath = DICOM_DIR + filename + "/"
    else:
        filepath = filename
        
    dicom_file_list = [f for f in listdir(filepath) if isfile(join(filepath, f))]
    dicom_image_data = []
    for dicom_file in dicom_file_list:
        dicom_filepath  = filepath + dicom_file
#       index, extension = dicom_file.split('.')
#       index = int(index)
        image_data = parse_dicom_file(dicom_filepath)
        dicom_image_data.append(image_data['pixel_data'])
    return np.array(dicom_image_data)  # 3 dimensional dicom data np array [image_no, pixel_width, pixel_height]

#http://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def process_dicoms(dicom_dir = DICOM_DIR):
    dicom_filenames = get_immediate_subdirectories(dicom_dir)
    # dicom_filenames = [d for d in listdir(dicom_dir) if isfile(join(dicom_dir, d))]
    dicom_data = {}
    for dicom_file in dicom_filenames:
        pixel_data = process_dicom_file(dicom_file)
        dicom_data[dicom_file] = pixel_data
    return dicom_data

def process_contours(contour_dir = CONTOUR_DIR, flag = 'inner'):
    CONTOUR_TYPE = 'icontour' if flag == 'inner' else 'ocontour'
    contour_filenames = get_immediate_subdirectories(contour_dir)
    contour_data = {}
    for contour_file in contour_filenames:
        mask_data = process_contour_file(contour_file)
        contour_data[contour_file] = mask_data
    return contour_data
# link dicoms to contours 

def link_dicoms_and_contours(dicom_data, contour_data, link_map):
    contour_matrix = np.array([])
    dicom_matrix = np.array([])
    
    for dicom_id in dicom_data:
        if dicom_id not in link_map:
            print "dicom file id %d does not have an entry in linkage map - skipping " % (dicom_id)
            continue
        contour_id = link_map[dicom_id]
        if contour_id not in contour_data:
            print "dicom_id: %d does not have a matching contour image data for id: %d" % (dicom_id, contour_id)
            continue
            
        patient_dicom_data = dicom_data[dicom_id].astype(int)
        num_dicom_images = len(patient_dicom_data)
        expanded_contours = np.expand_dims(contour_data[contour_id].astype(int), axis = 0)
        repeated_contours = np.repeat(expanded_contours, repeats = num_dicom_images, axis = 0) 
        
        contour_matrix = np.vstack([contour_matrix, repeated_contours]) if contour_matrix.size else repeated_contours
        dicom_matrix = np.vstack([dicom_matrix, patient_dicom_data]) if dicom_matrix.size else patient_dicom_data
        
    return (contour_matrix, dicom_matrix)
    
def create_data_pipeline(dicom_dir = DICOM_DIR, contour_dir = CONTOUR_DIR, link_file = LINK_FILE, flag = 'inner'):
    dicom_data = process_dicoms(dicom_dir)
    contour_data = process_contours(contour_dir)

    links = process_link_file(link_file)
    
    contour_matrix, dicom_matrix = link_dicoms_and_contours(dicom_data, contour_data, links)
    
    # match individual dicom images each with the same contour
    return contour_matrix, dicom_matrix

