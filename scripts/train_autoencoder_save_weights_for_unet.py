# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 21:03:12 2022

@author: manos
"""
import random
import numpy as np
import cv2
from keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, Input, Conv2DTranspose
from tensorflow.keras.models import Sequential
import os
from keras.models import Model
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
import pydicom
from preprocessing import crop_and_pad
from models.autoencoder_model import build_autoencoder, build_encoder
from models.unet_model import build_unet

SIZE=256


""" Read images """
dataset_path = os.path.join('..', 'dataset')

images = sorted(glob('../dataset/diana_segmented/**/*.IMA', recursive=True))
images += sorted(glob('../dataset/aorte_segmented/**/*.ima', recursive=True))
images += sorted(glob('../dataset/marfan_segmented/**/*.ima', recursive=True))


img_data=[]
for i in tqdm(images):
    dcm = pydicom.dcmread(i)
    x = dcm.pixel_array
    x = crop_and_pad(x, SIZE, SIZE)
    x = x/np.max(x)
    x = x.astype(np.float32)
    img_data.append(x)
    
img_array = np.reshape(img_data, (len(img_data), SIZE, SIZE, 1))
# img_array = img_array.astype('float32') / 255.

""" Define the autoencoder model """
autoencoder_model=build_autoencoder((SIZE, SIZE, 1))
autoencoder_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
print(autoencoder_model.summary())

""" Train the autoencoder """
history = autoencoder_model.fit(img_array, img_array,
        epochs=100, verbose=1)

autoencoder_model.save('autoencoder.h5')

""" Test on a few images """       
num=random.randint(0, len(img_array)-1)
test_img = np.expand_dims(img_array[num], axis=0)
pred = autoencoder_model.predict(test_img)

plt.subplot(1,2,1)
plt.imshow(test_img[0])
plt.title('Original')
plt.subplot(1,2,2)
plt.imshow(pred[0].reshape(SIZE,SIZE,3))
plt.title('Reconstructed')
plt.show()


""" Set weights to encoder part of the U-net (first 35 layers) """
unet_model = build_unet((SIZE, SIZE, 1))

for l1, l2 in zip(unet_model.layers[:35], autoencoder_model.layers[0:35]):
    l1.set_weights(l2.get_weights())

unet_model.save('unet_pretrained.h5')