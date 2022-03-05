# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 14:02:30 2022

@author: manos
"""
import numpy as np
import tensorflow as tf
import os
from natsort import natsorted
from glob import glob

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)
        
def train_val_test_split(images, masks, split):
    
    image_names = [i.split(os.sep)[-1] for i in images]
    image_names = np.unique([i.split('_')[0] for i in image_names])
    
    mask_names = [i.split(os.sep)[-1] for i in masks]
    mask_names = np.unique([i.split('_')[0] for i in mask_names])
    
    split_size = int(len(image_names) * split)
    splitted_ids = np.split(image_names, [split_size, 2*split_size])
    
    """ Get training dataset """
    images_train = [i for i in images if i[18:].split('_')[0] in splitted_ids[2]]
    masks_train = [i for i in masks if i[17:].split('_')[0] in splitted_ids[2]]
    
    """ Get validation dataset """
    images_valid = [i for i in images if i[18:].split('_')[0] in splitted_ids[1]]
    masks_valid = [i for i in masks if i[17:].split('_')[0] in splitted_ids[1]]
    
    """ Get test dataset """
    images_test = [i for i in images if i[18:].split('_')[0] in splitted_ids[0]]
    masks_test = [i for i in masks if i[17:].split('_')[0] in splitted_ids[0]]

    return (images_train, masks_train), (images_valid, masks_valid), (images_test, masks_test)

if __name__ == "__main__":

    """ Split unique names into three """
    split = 0.2

    """ Dataset """
    dataset_path = os.path.join('..', 'dataset')
    
    # """ Directory for storing files """
    # train_path = os.path.join(dataset_path, 'train')
    # valid_path = os.path.join(dataset_path, 'valid')
    # test_path = os.path.join(dataset_path, 'test')
    
    # """ Create new directories if doesn't exist """
    # create_dir(train_path)
    # create_dir(valid_path)
    # create_dir(test_path)
    
    """ Sort images based on patient ID """
    images = natsorted(glob(os.path.join(dataset_path, "images", "*.dcm")))
    masks = natsorted(glob(os.path.join(dataset_path, "masks", "*.png")))
    
    (images_train, masks_train), (images_valid, masks_valid), (images_test, masks_test) = train_val_test_split(images, masks, split)