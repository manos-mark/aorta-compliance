# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 21:03:12 2022

@author: manos
"""
from tensorflow.keras.optimizers import Adam

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import random
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Conv2DTranspose
from tensorflow.keras.models import Sequential
from keras.models import Model
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
import pydicom
from preprocessing import crop_and_pad, contrast_stretching
from models.autoencoder_model import build_autoencoder, build_encoder
from models.unet_model import build_unet
import tensorflow as tf
import pydicom as dicom
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

H=W=256

def read_image(path):
    dcm = dicom.dcmread(path)
    x = dcm.pixel_array
    x = contrast_stretching(x)
    x = crop_and_pad(x, W, H)
    x = x/np.max(x)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x):
    def _parse(x, y):
        x = x.decode()
        x = read_image(x)
        return x, x

    x, y = tf.numpy_function(_parse, [x, x], [tf.float32, tf.float32])
    x.set_shape([H, W, 1])
    y.set_shape([H, W, 1])
    
    return x, y

def tf_dataset(X, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((X))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    """ Read images """
    dataset_path = os.path.join('..', 'dataset')
    
    images = sorted(glob('../dataset/diana_segmented/**/*.IMA', recursive=True))
    # images = images[:100]
    # images += sorted(glob('../dataset/aorte_segmented/**/*.ima', recursive=True))
    # images += sorted(glob('../dataset/marfan_segmented/**/*.ima', recursive=True))
    
    # img_data=[]
    # for i in tqdm(images):
    #     dcm = pydicom.dcmread(i)
    #     x = dcm.pixel_array
    #     x = contrast_stretching(x)
    #     x = crop_and_pad(x, W, H)
    #     x = x/np.max(x)
    #     x = x.astype(np.float32)
    #     img_data.append(x)
        
    # img_array = np.reshape(img_data, (len(img_data), W, H, 1))
    # img_array = img_array.astype('float32') / 255.
    
    dataset = tf_dataset(images, batch=16)
    
    """ Define the autoencoder model """
    autoencoder_model=build_autoencoder((W, H, 1))
    autoencoder_model.compile(optimizer=Adam(0.01), loss='mean_squared_error', metrics=['accuracy'])
    print(autoencoder_model.summary())
    
    """ Train the autoencoder """
    history = autoencoder_model.fit(dataset,
            epochs=100, verbose=1)
    
    autoencoder_model.save('autoencoder.h5')
    
    """ Set weights to encoder part of the U-net (first 35 layers) """
    unet_model = build_unet((W, H, 1))
    
    for l1, l2 in zip(unet_model.layers[:35], autoencoder_model.layers[0:35]):
        l1.set_weights(l2.get_weights())
    
    unet_model.save('unet_pretrained.h5')