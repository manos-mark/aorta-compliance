
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
import pydicom as dicom
from natsort import natsorted
from keras.preprocessing.image import ImageDataGenerator

from scripts.metrics import dice_loss, dice_coef, iou
from scripts.unet_model import build_unet

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
""" Global parameters """
H = 256
W = 256

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, split=0.1):
    images = natsorted(glob(os.path.join(path, "images", "*.dcm")))
    masks = natsorted(glob(os.path.join(path, "masks", "*.png")))

    split_size = int(len(images) * split)

    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=split_size, random_state=42)
    # train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=split_size, random_state=42)
    # train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    dcm = dicom.dcmread(path)
    dcm = dcm.pixel_array
    # x = cv2.imread(path, cv2.IMREAD_COLOR)
    # x = np.array(dcm)
    # x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(dcm, (W, H))
    x = x/np.max(x)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (W, H))
    x = x/np.max(x)
    x = x > 0.5
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = x.decode()
        y = y.decode()

        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 1])
    y.set_shape([H, W, 1])
    print(x.shape)
    return x, y

def tf_dataset(X, Y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(4)
    
    print(np.array(dataset.as_numpy_iterator()).nbytes)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)#set_random_seed(42)

    """ Directory for storing files """
    create_dir("output")

    """ Hyperparameters """
    batch_size = 2
    lr = 1e-5
    num_epochs = 200
    model_path = os.path.join("output", "model.h5")
    csv_path = os.path.join("output", "data.csv")

    """ Dataset """
    dataset_path = os.path.join('.', 'dataset')
    
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)
    
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test: {len(test_x)} - {len(test_y)}")
    
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)
    
    """ Data augmentation layers """
    data_augmentation = None
    # data_augmentation = tf.keras.Sequential([
    #   tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    #     tf.keras.layers.experimental.preprocessing.RandomRotation(0.2), 
    #    tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=(0.2, 0.3), width_factor=(0.2, 0.3)),
    #    tf.keras.layers.experimental.preprocessing.RandomTranslation(0.3, 0.3, fill_mode='reflect', interpolation='bilinear',)
    # ])
    
    """ Model """
    model = build_unet((H, W, 1), data_augmentation)
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
    #     plt.subplot(1,3,3)
    #     plt.hist(image[:,:,0])
    #     plt.show()

    # model.summary()

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path)
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )