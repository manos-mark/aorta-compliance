
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

from sklearn.model_selection import KFold
from keras_unet_collection import models
from keras_unet_collection.losses import dice
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
import pydicom as dicom
from natsort import natsorted
import datetime

from metrics import dice_loss, dice_coef, iou, hausdorff
from models.unet_model import build_unet
from models.res_unet_model import build_res_unet
from models.attention_unet_model import build_attention_unet
from models.attention_res_unet_model import build_attention_res_unet
import datetime
from utils import *


""" Global parameters """
H = 256
W = 256
EXPERIMENT = "unet-diana_healthy_marfan-lr_0.001-batch_1-augmented"

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("output")

    """ Hyperparameters """
    batch_size = 1
    lr = 1e-3
    num_epochs = 200

    """ Dataset """
    dataset_path = os.path.join('..', 'dataset')
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path, split=0.2)
    # (train_x, train_y), (test_x, test_y) = load_data(dataset_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size, augment=True)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size, augment=False)

    """ Model """
    # model = models.unet_2d((128, 128, 3), [64, 128, 256, 512, 1024], n_labels=1,
    #                     stack_num_down=2, stack_num_up=2,
    #                     activation='ReLU', output_activation='Sigmoid', 
    #                     backbone='VGG16', weights='imagenet', 
    #                     freeze_backbone=True, freeze_batch_norm=True,
    #                     batch_norm=False, pool='max', unpool=False, name='unet')

    # model = models.transunet_2d((H, W, 1), filter_num=[64, 128, 256, 512], n_labels=1, stack_num_down=2, stack_num_up=2,
    #                             embed_dim=768, num_mlp=3072, num_heads=1, num_transformer=1,
    #                             activation='ReLU', mlp_activation='GELU', output_activation='Sigmoid', 
    #                             batch_norm=True, pool=True, unpool=False, name='transunet')

    model = build_unet((H,W,1))
    # pretrained_model_path = os.path.join('..', 'output', 
    #     'autoencoder-batch_16-epochs_500-adam-mse-relu', 'unet_pretrained.h5') 
    # model.load_weights(pretrained_model_path)
    # for l1 in model.layers[:14]:
    #     l1.trainable = False

    metrics = [dice_coef, iou, hausdorff, Precision(), Recall()]
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=metrics)
    # model.summary()
    
    log_dir = os.path.join('..', 'logs', EXPERIMENT, 'fit', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    model_path = os.path.join("..", "output", EXPERIMENT, "model.h5")
    csv_path = os.path.join("..", "output", EXPERIMENT, "data.csv")
    callbacks = [
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=16),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    
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