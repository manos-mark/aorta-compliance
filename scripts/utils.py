import pydicom
import numpy as np
from natsort import natsorted
import os
from glob import glob
from preprocessing import contrast_stretching, crop_and_pad
import tensorflow as tf
import cv2

""" Global parameters """
H = 256
W = 256

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

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.2):
    images = natsorted(glob(os.path.join(path, "images", "*.dcm")))
    masks = natsorted(glob(os.path.join(path, "masks", "*.png")))
    return train_val_test_split(images, masks, split)

def read_image(path):
    dcm = pydicom.dcmread(path)
    x = dcm.pixel_array
    x = contrast_stretching(x)
    x = crop_and_pad(x, W, H)
    x = x/np.max(x)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = crop_and_pad(x, W, H)
    x = x/np.max(x)
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 1])
    y.set_shape([H, W, 1])
    
    return x, y

def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset