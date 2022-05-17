import pydicom
import numpy as np
from natsort import natsorted
import os
from glob import glob
from preprocessing import contrast_stretching, crop_and_pad, bm3d_denoising
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
from albumentations import (
    Compose, RandomBrightnessContrast, HorizontalFlip,
    Rotate, Affine, VerticalFlip
)
from functools import partial

""" Global parameters """
H = 256
W = 256

def train_val_test_split(images, masks, split):
    
    image_names = [i.split(os.sep)[-1] for i in images]
    image_names = np.unique([i.split('_')[0] for i in image_names])

    np.random.seed(0)
    np.random.shuffle(image_names)

    split_size = int(len(image_names) * split)
    splitted_ids = np.split(image_names, [split_size, 2*split_size])
    
    """ Get training dataset """
    images_train = np.array([i for i in images if i[18:].split('_')[0] in splitted_ids[2]])
    masks_train = np.array([i for i in masks if i[17:].split('_')[0] in splitted_ids[2]])
    
    """ Get validation dataset """
    images_valid = [i for i in images if i[18:].split('_')[0] in splitted_ids[1]]
    masks_valid = [i for i in masks if i[17:].split('_')[0] in splitted_ids[1]]
    
    """ Get test dataset """
    images_test = np.array([i for i in images if i[18:].split('_')[0] in splitted_ids[0]])
    masks_test = np.array([i for i in masks if i[17:].split('_')[0] in splitted_ids[0]])

    return (images_train, masks_train), (images_valid, masks_valid), (images_test, masks_test)


def train_test_split(images, masks, split):
    image_names = [i.split(os.sep)[-1] for i in images]
    image_names = np.unique([i.split('_')[0] for i in image_names])

    np.random.seed(0)
    np.random.shuffle(image_names)

    split_size = int(len(image_names) * split)
    splitted_ids = np.split(image_names, [split_size])
    
    """ Get training dataset """
    images_train = np.array([i for i in images if i[18:].split('_')[0] in splitted_ids[1]])
    masks_train = np.array([i for i in masks if i[17:].split('_')[0] in splitted_ids[1]])
    
    """ Get test dataset """
    images_test = np.array([i for i in images if i[18:].split('_')[0] in splitted_ids[0]])
    masks_test = np.array([i for i in masks if i[17:].split('_')[0] in splitted_ids[0]])

    return (images_train, masks_train), (images_test, masks_test)

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def data_augmentation(image, mask, img_size):
    def _data_augmentation(image, mask, img_size):
        transforms = Compose([
            Rotate(limit=20),
            RandomBrightnessContrast(),
            HorizontalFlip(),
            VerticalFlip(),
            Affine(scale=(0.9, 1.2)),

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
    # images = natsorted(glob(os.path.join(path, "images", "*.dcm")))
    images = natsorted(glob(os.path.join(path, "images", "*.npy")))
    if len(images) == 0:
        images = natsorted(glob(os.path.join(path, "images", "*.nii.gz")))
    # masks = natsorted(glob(os.path.join(path, "masks", "*.png")))
    masks = natsorted(glob(os.path.join(path, "masks", "*.npy")))
    
    return train_test_split(images, masks, split)

def read_image(path):
    # dcm = pydicom.dcmread(path)
    # x = dcm.pixel_array
    # x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = np.load(path)
    x = contrast_stretching(x)
    x = crop_and_pad(x, H, W)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)

    # 2.5D architecture 
    # x = nib.load(path)
    # x = x.get_data()
    # x = contrast_stretching(x)
    # x = crop_and_pad(x, H, W)
    # new_x = np.zeros((H,W,3))
    # for i in range(x.shape[2]):
    #     new_x[:,:,i] = crop_and_pad(x[:,:,i], H, W)
    # new_x = (new_x - np.min(new_x)) / (np.max(new_x) - np.min(new_x))
    # new_x = new_x.astype(np.float32)

    
    return x

def read_mask(path):
    # x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = np.load(path)
    x = crop_and_pad(x, H, W)
    # x = x/np.max(x)
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
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

def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.title('Bland-Altman Plot')
    plt.show()