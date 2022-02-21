# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 21:43:58 2022

@author: manos
"""

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow.keras.layers
from tensorflow import keras

def conv_block(input, num_filters):
    conv = Conv2D(num_filters, 3, padding="same")(input)
    conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(num_filters, 3, padding="same")(conv)
    conv = BatchNormalization(axis=3)(conv)
    
    shortcut = Conv2D(num_filters, 1, padding="same")(input)
    shortcut = BatchNormalization(axis=3)(shortcut)
    
    res_path = tensorflow.keras.layers.add([shortcut, conv])
    res_path = Activation("relu")(res_path) 

    return res_path

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)
    return x, p

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_res_unet(input_shape, preprocessing=None):
    inputs = Input(input_shape)
    
    if type(preprocessing) == keras.Sequential:
        inputs = preprocessing(inputs)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b1 = conv_block(p4, 1024)

    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="U-Net")
    return model

if __name__ == "__main__":
    input_shape = (512, 512, 3)
    model = build_unet(input_shape)
    model.summary()
