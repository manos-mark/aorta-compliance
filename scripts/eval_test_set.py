import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import datetime
import numpy as np
from tensorflow.keras.callbacks import CSVLogger
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
import pydicom 
from preprocessing import crop_and_pad
import matplotlib.pyplot as plt
import pandas as pd
from metrics import dice_loss, dice_coef, iou, hausdorff, focal_tversky_loss
from train import load_data, create_dir, tf_dataset, read_image, read_mask
from utils import create_dir

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
    
""" Global parameters """
H = 256
W = 256
EXPERIMENT = 'unet-diana-focal_tversky_loss-lr_0.01-batch_8-augmented-healthy'
OUTPUT_FOLDER_PATH = os.path.join('..', 'results', EXPERIMENT)

if __name__ == "__main__":
    # interpret_training_results()
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")
    create_dir(OUTPUT_FOLDER_PATH)

    """ Dataset """
    dataset_path = os.path.join('..', 'dataset')
    
    (_, _), (_, _), (test_x, test_y) = load_data(dataset_path, split=0.2)
    
#    print(f"Train: {len(train_x)} - {len(train_y)}")
#    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    test_dataset = tf_dataset(test_x, test_y, batch=1, augment=False)
    

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'focal_tversky_loss': focal_tversky_loss, 'hausdorff': hausdorff}):
        model = tf.keras.models.load_model(os.path.join('..', "output", EXPERIMENT, "model.h5"))
    
    callbacks = [
        CSVLogger(os.path.join("..", "output", EXPERIMENT, "test.csv"))
    ]

    h = model.evaluate(test_dataset, batch_size=2, callbacks=callbacks, verbose=1)
    
    print(h)
    
    dice_cores = []
    iou_cores = []
    hd_scores = []
    for i, (x,y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
         H, W = ((pydicom.dcmread(x)).pixel_array).shape
         x = read_image(x)
         y = read_mask(y)
        
        
         """ Predicting the mask """
        #  y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        #  y_pred = y_pred.astype(np.float32)
         # y_pred = crop_and_pad(y_pred[:,:,0], W, H)
         # x = crop_and_pad(x[:,:,0], W, H)
         
         scores = model.evaluate(np.expand_dims(x, axis=0),np.expand_dims(y, axis=0), verbose=0)
        #  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
         dice_cores.append(scores[1] * 100)
         iou_cores.append(scores[2] * 100)
         hd_scores.append(scores[3])
     
    print(model.metrics_names[1], " %.2f%% (+/- %.2f%%)" % (np.mean(dice_cores), np.std(dice_cores)))
    print(model.metrics_names[2], " %.2f%% (+/- %.2f%%)" % (np.mean(iou_cores), np.std(iou_cores)))
    print(model.metrics_names[3], " %.2f (+/- %.2f)" % (np.mean(hd_scores), np.std(hd_scores)))

#         fig = plt.figure(figsize=(15,15))
#         plt.subplot(2,2,1, title='Gound truth mask')
#         plt.imshow(y, cmap='gray')
#         plt.subplot(2,2,2, title='Image and ground truth mask')
#         plt.imshow(x, cmap='gray')
#         plt.imshow(y, cmap='jet', alpha=0.2)
#         
#         plt.subplot(2,2,3, title='Predicted mask')
#         plt.imshow(y_pred, cmap='gray')
#         plt.subplot(2,2,4, title='Image and predicted mask')
#         plt.imshow(x, cmap='gray')
#         plt.imshow(y_pred, cmap='jet', alpha=0.2)
#         plt.tight_layout()
        
    #     """ Saving the predicted mask along with the image and GT """
    #     save_image_path = os.path.join(OUTPUT_FOLDER_PATH, str(i))
    #     plt.savefig(save_image_path)
        