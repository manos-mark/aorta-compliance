import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from scripts.metrics import dice_loss, dice_coef, iou
from scripts.train import load_data, create_dir, tf_dataset, read_image, read_mask
import pydicom as dicom
from itertools import islice
import matplotlib.pyplot as plt
import pandas as pd

H = 256
W = 256

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(os.path.join(".", "output", "model.h5"))

    """ Dataset """
    dataset_path = os.path.join('.', 'dataset')
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    test_dataset = tf_dataset(test_x, test_y, batch=1)
    
    dataset_list = list(test_dataset)
    # dataset_list = np.array(test_dataset.as_numpy_iterator())
    slices = tf.data.Dataset.from_tensor_slices(dataset_list)
    
    
    dices  = []
    ious = []
    for i, sl in enumerate(slices):
        test_image, test_mask = next(islice(slices, i, None))
        

        # for j in range(0, 1):
            # test_image = test_image_iter[j]
            # test_mask = test_mask_iter[j]
            
        """ Extracing the image name. """
        # image_name = test_image.split("/")[-1]
        
        """ Predicting the mask """
        y_pred = model.predict(test_image)[0] > 0.5
        y_pred = y_pred.astype(np.float32)
        
        
        # for num in tqdm(nums):
        dices.append(dice_coef(np.expand_dims(test_mask, axis=0), np.expand_dims(y_pred, axis=0)))
        ious.append(iou(test_mask, y_pred))
        
        # plt.subplot(2,2,1, title='Gound truth mask')
        # plt.imshow(test_mask, cmap='gray')
        # plt.subplot(2,2,2, title='Gound truth image & mask')
        # plt.imshow(test_image, cmap='gray')
        # plt.imshow(test_mask, cmap='jet', alpha=0.2)
        
        # plt.subplot(2,2,3, title='Predicted mask')
        # plt.imshow(y_pred, cmap='gray')
        # plt.subplot(2,2,4, title='Predicted image & mask')
        # plt.imshow(test_image, cmap='gray')
        # final = plt.imshow(y_pred, cmap='jet', alpha=0.2)
        # plt.tight_layout()
        
        # """ Saving the predicted mask along with the image and GT """
        # save_image_path = os.path.join('.', 'results', str(i))
        # plt.savefig(save_image_path)

        # plt.show()
    
    print("Test Dice Coef mean: ", np.mean(np.array(dices)))
    print("Test IoU mean: ", np.mean(np.array(ious)))
            
    log_data = pd.read_csv(os.path.join('.', 'output', 'data.csv'))
    
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
    

    """ Predicting the mask """
    # for x_path, y_path in tqdm(zip(test_x, test_y), total=len(test_x)):
    #     """ Extracing the image name. """
    #     image_name = x_path.split("/")[-1]

    #     """ Reading the image """
    #     x = dicom.dcmread(x_path)
    #     x = x.pixel_array
    #     x = cv2.resize(x, (W, H))
    #     x = x/np.max(x)
    #     x = x.astype(np.float32)
    #     x = np.expand_dims(x, axis=-1)  ## (256, 256, 1)
    #     x = np.expand_dims(x, axis=0)  ## (1, 256, 256, 1)

    #     """ Reading the mask """
    #     y = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
    #     y = cv2.resize(y, (W, H))
    #     y = np.expand_dims(y, axis=-1)  ## (256, 256, 1)
    #     y = np.expand_dims(y, axis=0)   ## (1, 256, 256, 1)
    #     # y = np.concatenate([y, y, y], axis=-1)  ## (256, 256, 3)

    #     """ Predicting the mask. """
    #     y_pred = model.predict(x)[0] > 0.5
    #     # y_pred = model.predict(x)[0] > 0.5
    #     y_pred = y_pred.astype(np.int32)

        # """ Saving the predicted mask along with the image and GT """
        # save_image_path = os.path.join('.', 'results', image_name)
        # # y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

        # # sep_line = np.ones((H, 10, 3)) * 255

        # # cat_image = np.concatenate([x, sep_line, y, sep_line, y_pred*255], axis=1)
        # cv2.imwrite(save_image_path, plt.volume)
