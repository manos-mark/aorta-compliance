# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 21:03:12 2022

@author: manos
"""

import os
import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import numpy as np
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Conv2DTranspose
from tensorflow.keras.models import Sequential
from keras.models import Model
from glob import glob
from models.autoencoder_model import build_autoencoder
from models.unet_model import build_unet
from models.attention_res_unet_model import build_attention_res_unet
import tensorflow as tf
from utils import read_image, load_data
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def tf_parse(x):
    def _parse(x):
        x = x.decode()
        x = read_image(x)
        return x

    x = tf.numpy_function(_parse, [x], tf.float32)
    x.set_shape([H, W, 1])
    # y.set_shape([H, W, 1])
    
    return x, x

def tf_dataset(X, batch=1):
    dataset = tf.data.Dataset.from_tensor_slices((X))
    dataset = dataset.cache()
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

""" Global parameters """
H=W=256
EXPERIMENT = "autoencoder-batch_16-epochs_500-adam-mse-relu-AORTE_TRAIN"

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Hyperparameters """
    batch_size = 16
    # lr = 1e-3
    num_epochs = 500
    loss = 'mean_squared_error' # or 'abs_squared_error'
    optimizer = 'adam' # or 'adam'

    """ Read images """
    dataset_path = os.path.join('..', 'dataset')

    # images = sorted(glob('../dataset/diana_segmented/**/*.IMA', recursive=True))
    (train_x, _), (_, _), (_, _) = load_data(dataset_path)
    dataset = tf_dataset(train_x, batch=16)
    
    """ Define the autoencoder model """
    autoencoder_model=build_autoencoder((W, H, 1))
    # autoencoder_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse','mae'])
    autoencoder_model.compile(optimizer=optimizer, loss=loss, metrics=['mse','mae'])
    print(autoencoder_model.summary())
    
    """ Train the autoencoder """
    log_dir = os.path.join('..', 'logs', EXPERIMENT, 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_path = os.path.join("..", "output", EXPERIMENT, "model.h5")

    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='loss', patience=10),
    ]

    history = autoencoder_model.fit(
            dataset,
            epochs=num_epochs, 
            shuffle=True,
            callbacks=callbacks)

    autoencoder_path = os.path.join('..', 'output', EXPERIMENT, 'autoencoder.h5')
    autoencoder_model.save(autoencoder_path)
        
    """ Set weights to encoder part of the U-net (first 35 layers) """
    unet_model = build_unet((W, H, 1))
    
    for l1, l2 in zip(unet_model.layers[:35], autoencoder_model.layers[0:35]):
        l1.set_weights(l2.get_weights())
    
    unet_model.save(os.path.join('..', 'output', EXPERIMENT, 'unet_pretrained.h5'))