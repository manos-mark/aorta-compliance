import pydicom
import numpy as np
from natsort import natsorted
import os
from glob import glob
from preprocessing import contrast_stretching, crop_and_pad
import tensorflow as tf
import cv2
import albumentations
from albumentations import (
    Compose, RandomBrightness, JpegCompression, HueSaturationValue, RandomContrast, HorizontalFlip,
    Rotate, Affine
)
from functools import partial

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

def data_augmentation(image, mask, img_size):
    def _data_augmentation(image, mask, img_size):
        transforms = Compose([
            Rotate(limit=20),
            # RandomBrightness(limit=0.1),
            # JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
            # HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            # RandomContrast(limit=0.2, p=0.5),
            HorizontalFlip(),
            # Affine()
        ])
        data = {"image":image, "mask":mask}
        aug_data = transforms(**data)
        aug_img = aug_data["image"]
        aug_mask = aug_data["mask"]
        # aug_img = tf.cast(aug_img/255.0, tf.float32)
        # aug_img = tf.image.resize(aug_img, size=[img_size, img_size])
        return aug_img, aug_mask

    aug_img, aug_mask = tf.numpy_function(func=_data_augmentation, inp=[image, mask, img_size], Tout=[tf.float32,tf.float32])
    # aug_img.set_shape([H, W, 1])
    # aug_mask.set_shape([H, W, 1])
    return aug_img, aug_mask

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

def tf_dataset(X, Y, batch=2, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    if augment:
        dataset = dataset.map(partial(data_augmentation, img_size=W))
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset