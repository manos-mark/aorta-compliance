import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import pydicom as dicom
from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd

from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir, tf_dataset, read_image, read_mask


H = 256
W = 256
EXPERIMENT = 'attention-u-net_not-augmented_split-patients'

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def interpret_training_results():
    log_data = pd.read_csv(os.path.join('..', 'output', EXPERIMENT, 'data.csv'))
    
    log_data[['epoch', 'dice_coef', 'val_dice_coef']].plot(
        x='epoch',
        xlabel='Epochs',
        ylabel='Value',
        title='Dice Coef vs Validation Dice Coef'
    )
    
    log_data[['epoch', 'iou', 'val_iou']].plot(
        x='epoch',
        xlabel='Epochs',
        ylabel='Value',
        title='IoU vs Val IoU'
    )
    
    log_data[['epoch', 'loss', 'val_loss']].plot(
        x='epoch',
        xlabel='Epochs',
        ylabel='Value',
        title='Loss vs Validation Loss'
    )
    
    log_data[['epoch', 'precision', 'val_precision']].plot(
        x='epoch',
        xlabel='Epochs',
        ylabel='Value',
        title='Precision vs Validation Precision'
    )
    
    log_data[['epoch', 'recall', 'val_recall']].plot(
        x='epoch',
        xlabel='Epochs',
        ylabel='Value',
        title='Recall vs Validation Recall'
    )
    

if __name__ == "__main__":
    # interpret_training_results()
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Dataset """
    dataset_path = os.path.join('..', 'dataset')
    
    (_, _), (_, _), (test_x, test_y) = load_data(dataset_path)
    
#    print(f"Train: {len(train_x)} - {len(train_y)}")
#    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    test_dataset = tf_dataset(test_x, test_y, batch=1)
    

    """ Loading model """
    from focal_loss import BinaryFocalLoss
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'BinaryFocalLoss': BinaryFocalLoss(gamma=0.2)}):
        model = tf.keras.models.load_model(os.path.join('..', "output", EXPERIMENT, "model.h5"))
    
    model.evaluate(test_dataset, batch_size=2)
    
    
    for i, (x,y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        x = read_image(x)
        y = read_mask(y)
        
        """ Extracing the image name. """
        # image_name = test_image.split("/")[-1]
        
        """ Predicting the mask """
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        y_pred = y_pred.astype(np.float32)

        # plt.subplot(2,2,1, title='Gound truth mask')
        # plt.imshow(y, cmap='gray')
        # plt.subplot(2,2,2, title='Gound truth image & mask')
        # plt.imshow(x, cmap='gray')
        # plt.imshow(y, cmap='jet', alpha=0.2)
         
        # plt.subplot(2,2,3, title='Predicted mask')
        # plt.imshow(y_pred, cmap='gray')
        # plt.subplot(2,2,4, title='Predicted image & mask')
        # plt.imshow(x, cmap='gray')
        # final = plt.imshow(y_pred, cmap='jet', alpha=0.2)
        # plt.tight_layout()

        plt.subplot(title='Predicted image & mask')
        plt.imshow(x, cmap='gray')
        final = plt.imshow(y_pred, cmap='jet', alpha=0.2)
        
        """ Saving the predicted mask along with the image and GT """
        create_dir(os.path.join('..', 'results', EXPERIMENT))
        save_image_path = os.path.join('..', 'results', EXPERIMENT, str(i))
        plt.savefig(save_image_path)
        
#        plt.show()