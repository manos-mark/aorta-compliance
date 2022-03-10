
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from scipy.spatial.distance import directed_hausdorff
smooth = 1e-15 

def hausdorff(y_true, y_pred):
    def f(y_true, y_pred, seed=0):
        y_true = tf.keras.layers.Flatten()(y_true)
        y_pred = tf.keras.layers.Flatten()(y_pred)
        return directed_hausdorff(y_true, y_pred, seed)[0]
    return tf.numpy_function(f, [y_true, y_pred], tf.double)


def iou(y_true, y_pred):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + smooth) / (union + smooth)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)

def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
#########################################################

from tensorflow.keras import backend as K
from sklearn.metrics import jaccard_score,confusion_matrix

def accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true.flatten(),y_pred.flatten(), labels=[0, 1])
    acc = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    return acc

# def iou(y_true, y_pred):
#     print(y_true, y_pred)
#     print(type(y_true), type(y_pred))
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

# def iou_loss(y_true, y_pred):
#     return -iou(y_true, y_pred)

#########################################################
# import tensorflow as tf
# from tensorflow.keras import models, layers, regularizers
# from tensorflow.keras import backend as K



# '''
# A few useful metrics and losses
# '''

# def dice_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


# def jacard_coef(y_true, y_pred):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


# def jacard_coef_loss(y_true, y_pred):
#     return -jacard_coef(y_true, y_pred)


# def dice_coef_loss(y_true, y_pred):
#     return -dice_coef(y_true, y_pred)
