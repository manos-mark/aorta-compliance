
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from tf_notification_callback import TelegramCallback
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
import pydicom as dicom
from natsort import natsorted
from keras.preprocessing.image import ImageDataGenerator
import skimage.transform
import datetime

from split_dataset import train_val_test_split
from metrics import dice_loss, dice_coef, iou
from models.unet_model import build_unet
from models.res_unet_model import build_res_unet
from models.attention_unet_model import build_attention_unet
from models.attention_res_unet_model import build_attention_res_unet
from preprocessing import augment, crop_and_pad, limiting_filter, contrast_stretching
import datetime
from utils import *

""" Global parameters """
H = 256
W = 256
EXPERIMENT = "att-res-u-net_lr_0.001-batch_8-dice_loss"

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("output")

    """ Hyperparameters """
    batch_size = 8
    lr = 1e-3
    num_epochs = 200

    model_path = os.path.join("..", "output", EXPERIMENT, "model.h5")
    csv_path = os.path.join("..", "output", EXPERIMENT, "data.csv")

    """ Dataset """
    dataset_path = os.path.join('..', 'dataset')
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    
    """ Data augmentation layers """
    data_augmentation = None
    # data_augmentation = tf.keras.Sequential([
    #     tf.keras.layers.RandomFlip("horizontal"),
    #     tf.keras.layers.RandomRotation(0.2), 
    #     tf.keras.layers.RandomZoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3)),
    #     tf.keras.layers.RandomTranslation(0.3, 0.3, fill_mode='reflect', interpolation='bilinear',)
    # ])
    
    """ Model """
    from focal_loss import BinaryFocalLoss
    model = build_attention_res_unet((H, W, 1))
    # pre_trained_unet_model.load_weights('unet_model_pretrained.h5')
    metrics = [dice_coef, iou, Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)
    
    """ Preview a random image and mask after processing """
    # from itertools import islice, count
    # import matplotlib.pyplot as plt
    # import random 
    # rand = random.randint(0, len(train_dataset))
    # train_image_iter, train_mask_iter = next(islice(train_dataset, rand, None))
    # for i in range(0, 1):
    #     image = train_image_iter[i]
    #     mask = train_mask_iter[i]
    #     plt.subplot(1,3,1)
    #     plt.imshow(image, cmap='gray')
    #     plt.subplot(1,3,2)
    #     plt.imshow(mask, cmap='gray')
    #     plt.show()

    model.summary()
    
    start = datetime.datetime.now()
    log_dir = os.path.join('..', 'logs', EXPERIMENT, 'fit', start.strftime("%Y%m%d-%H%M%S"))

    telegram_callback = TelegramCallback('5175590478:AAHP5_dnqpimmBt83-Z5EF2KSV3k1cqcElo',
                                        '1939791669',
                                        'CNN Model',
                                        # ['loss', 'val_loss'],
                                        False)

    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True),
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=10),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    
    end = datetime.datetime.now()
    diff = end-start
    days, seconds = diff.days, diff.seconds
    hours = days * 24 + seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    
    print(f"Total training time: {hours} hours \t {minutes} minutes \t {seconds} seconds")