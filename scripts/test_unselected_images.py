# -*- coding: utf-8 -*-
"""
Created on Sat Feb 12 12:03:05 2022

@author: manos
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import cv2
import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from scripts.metrics import dice_loss, dice_coef, iou
import pydicom as dicom
from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd
from natsort import natsorted

H = 256
W = 256

def read_image(path):
    dcm = dicom.dcmread(path)
    x = dcm.pixel_array
    # x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = np.array(dcm)
    # x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x/np.max(x)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_image(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 1])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(4)
    
    return dataset

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(os.path.join(".", "output", "model.h5"))

    """ Dataset """
    test_x = natsorted(glob.iglob('dataset/test_data/**/*.dcm', recursive=True))
    
    test_dataset = tf_dataset(test_x, test_x, batch=1)
    
    dataset_list = list(test_dataset)
    # dataset_list = np.array(test_dataset.as_numpy_iterator())
    slices = tf.data.Dataset.from_tensor_slices(dataset_list)
    
    
    for i, sl in enumerate(slices):
        test_image, _ = next(islice(slices, i, None))
        
        """ Predicting the mask """
        y_pred = model.predict(test_image)[0] > 0.5
        y_pred = y_pred.astype(np.float32)
        
        plt.subplot(2,2,2, title='Gound truth image')
        plt.imshow(test_image[0], cmap='gray')
        
        plt.subplot(2,2,3, title='Predicted mask')
        plt.imshow(y_pred, cmap='gray')
        plt.subplot(2,2,4, title='Predicted image & mask')
        plt.imshow(test_image[0], cmap='gray')
        final = plt.imshow(y_pred, cmap='jet', alpha=0.2)
        plt.tight_layout()
        
        # """ Saving the predicted mask along with the image and GT """
        save_image_path = os.path.join('.', 'results', 'unselected_patients', str(i))
        plt.savefig(save_image_path)

        plt.show()
            
