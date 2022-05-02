import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_coef, iou, dice_loss, hausdorff
from preprocessing import contrast_stretching, crop_and_pad
import tensorflow_addons as tfa

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# Class to Apply SegNet model
class SegmentationNetwork():
    def __init__(self): # Initialize the status bar and place it in the GUI
        # Load trained model weights
        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss, 'hausdorff': hausdorff, 'tfa': tfa}):
            self.model = tf.keras.models.load_model(resource_path('model.h5'))
        
        # Create optimizer, metrics and compile the model
        metrics = [iou, dice_coef, tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        self.model.compile(loss=dice_loss, metrics=metrics)
    
    def get_segmentation(self, volume):
        img_tf = self.prepare_for_net(volume)
        # Model Inference
        results = self.model.predict(img_tf, batch_size=1)
        # Postprocess of the predicion
        results = self.recover_shape(volume, results)
        results = np.squeeze(results>0.1)*1.0
        return results

    def prepare_for_net(self, img): # Preprocessing needed before the Model Inference
        # slice_cnt = img.shape[0]   # Get number of slices
        # img_tf = np.empty((slice_cnt, 256, 256, 1)) # Create the 3-Channels volume
        
        # for i in range(slice_cnt):
        #     new_img = img[i,:,:,:]
        #     new_img = contrast_stretching(new_img)
        #     new_img = crop_and_pad(new_img[:,:,0], 256,256)
        #     new_img = (new_img-new_img.min()) / (new_img.max() - new_img.min())   # Normalize in the range [0, 1]
        #     new_img = new_img.astype(np.float32)
        #     img_tf[i,:,:,0] = new_img
        
        img_tf = np.empty((1, 256, 256, 1))
        new_img = contrast_stretching(img)
        new_img = crop_and_pad(new_img[:,:,0], 256,256)
        new_img = (new_img-new_img.min()) / (new_img.max() - new_img.min())   # Normalize in the range [0, 1]
        new_img = new_img.astype(np.float32)
        img_tf[0,:,:,0] = new_img
        return img_tf

    def recover_shape(self, init_array, new_array): # Recover Original matrix size
        # img_tf = np.empty((new_array.shape[0], init_array.shape[1], init_array.shape[2], 1))
        # for i in range(new_array.shape[0]):
        #     img_tf[i,:,:,0] = crop_and_pad(new_array[i,:,:,0], init_array.shape[1], init_array.shape[2])
        img_tf = np.empty((init_array.shape[0], init_array.shape[1], 1))
        img_tf[:,:,0] = crop_and_pad(new_array[0,:,:,0], init_array.shape[0], init_array.shape[1])
        
        return img_tf

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