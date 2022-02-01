# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 09:58:27 2020

@author: FAaro
"""
import os, sys
import numpy as np
from skimage.morphology.convex_hull import convex_hull_image
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model
from keras_unet.models import custom_unet
from skimage.morphology import convex_hull_image
from skimage.measure import label
from skimage import img_as_bool


# Class to Apply SegNet model
class SegNet():
    def __init__(self): # Initialize the status bar and place it in the GUI
        # Create model and load trained model weights
        self.model = custom_unet(
                        input_shape=(512, 512, 3),
                        use_batch_norm=False,
                        upsample_mode="simple",
                        num_layers=4,
                        num_classes=1,
                        filters=8,
                        dropout=0.0,
                        output_activation='sigmoid')
        # self.model.load_weights(resource_path('model2.h5'))
        # # Create optimizer, metrics and compile the model
        # metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        # self.model.compile(loss="binary_crossentropy", metrics=metrics)
        # Preprocess the Volume
    
    def get_segmentation(self, volume):
        im_tf = prepare_for_net(volume)
        # Model Inference
        results = self.model.predict(im_tf, batch_size=1)
        # Postprocess of the predicion
        results = recover_shape(volume, results)
        # Retrieve only the Blood Pool
        results = np.squeeze(results>0.5)*1.0
        img_bw = img_as_bool(results)
        labels = label(img_bw, return_num=False)
        maxCC_nobcg = labels == np.argmax(np.bincount(labels.flat, weights=img_bw.flat))
        results = maxCC_nobcg*1.0
        return results

def prepare_for_net(np_array): # Preprocessing needed before the Model Inference
   num_im = np_array.shape[0]   # Get number of slices
   w, h = np_array.shape[1:3]   # Get width and height
   im_tf = np.empty((num_im, 512, 512, 3)) # Create the 3-Channels volume
   for i in range(num_im):
       im_3 = np_array[i,:,:,:]
       #im = (im-im.min()) / (im.max() - im.min())   # Normalize in the range [0, 1]
       #im_3 = np.empty((w, h, 3))
       #im_3[:,:,0] = im_3[:,:,1] = im_3[:,:,2] = im
       im_3 = resize(im_3, (512,512,3), preserve_range=True) # Resizing
       im_tf[i,:,:,:] = im_3
   return im_tf

def recover_shape(init_array, new_array): # Recover Original matrix size
  or_size = init_array.shape
  or_size = or_size[0:3]
  new_array = resize(new_array, or_size+(1,), preserve_range=True, order=0)
  return new_array

# For development, considers the relative path in string
# When build into .exe using pyinstaller, changes the directory
# To the TEMP folder wherever the executable unpacks the necessary files
# NOTE: this is only needed when using files apart from .py programs
# In this program, the icon and TF model
def resource_path(relative):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative)