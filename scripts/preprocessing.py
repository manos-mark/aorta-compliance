# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 10:37:20 2022

@author: manos
"""
import random
import os
import pydicom
from natsort import natsorted
from glob import glob
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import bm3d
import SimpleITK as sitk
import cv2
from medpy.filter import IntensityRangeStandardization

def standardization(image):
    # Get brain mask
    mask = image == 0
    
    selem = morphology.disk(2)
    
    segmentation = morphology.dilation(mask, selem)
    labels, label_nb = ndimage.label(segmentation)

    mask = labels == 0
    
    mask = morphology.dilation(mask, selem)
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, selem)
    
    # Normalize
    mean = image[mask].mean()
    std = image[mask].std()
    
    normalized_image = (image - mean) / std
    
    return normalized_image

def bm3d_denoising(image):
    bm3d_cleaned = bm3d.bm3d(image, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return bm3d_cleaned

def n4_bias_field_correction(image):
    sitk_image = sitk.GetImageFromArray(image)
    sitk_mask = sitk.OtsuThreshold(sitk_image,0,1)
    
    casted_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    output = corrector.Execute(casted_image, sitk_mask)
    
    return sitk.GetArrayFromImage(output)

def intensity_range_standardization(images):
    irs = IntensityRangeStandardization()
    trained_model, transformed_images = irs.train_transform(images)
    return transformed_images

def remove_noise_from_image(image, display=True):
    mask = image <= 10
    
    selem = morphology.disk(2)
    
    # We have to use the mask to perform segmentation 
    # due to the fact that the original image is quite noisy
    segmentation = morphology.dilation(mask, selem)
    labels, label_nb = ndimage.label(segmentation)

    mask = labels == 0
    
    mask = morphology.dilation(mask, selem)
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, selem)
    
    clean_image = mask * image

    if display:
        plt.figure(figsize=(15, 2.5))
        plt.subplot(141)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(142)
        plt.imshow(segmentation, cmap='gray')
        plt.title('Background mask')
        plt.axis('off')

        plt.subplot(143)
        plt.imshow(mask, cmap='gray')
        plt.title('Mask')
        plt.axis('off')

        plt.subplot(144)
        plt.imshow(clean_image, cmap='gray')
        plt.title('Clean Image')
        plt.axis('off')
    
    return clean_image


IMAGES_PATH = os.path.join('dataset', 'images')
MASKS_PATH = os.path.join('dataset', 'masks')

images = [pydicom.read_file(IMAGES_PATH + os.sep + s) for s in natsorted(os.listdir(IMAGES_PATH))[-3:-1]]
masks = [cv2.imread(MASKS_PATH + os.sep + s, cv2.IMREAD_GRAYSCALE) for s in natsorted(os.listdir(MASKS_PATH))[-3:-1]]

rand = random.randint(0, len(images)-1)
image_before = images[rand].pixel_array

images = [n4_bias_field_correction(i.pixel_array) for i in images]
images = [bm3d_denoising(i) for i in images]
images = [standardization(i) for i in images]
# images = [intensity_range_standardization(i) for i in images]

image_after = images[rand]
mask = masks[rand]

plt.figure()
plt.imshow(image_before, cmap='gray')
plt.show()

plt.figure()
plt.hist(image_before)
plt.show()

plt.figure()
plt.imshow(image_after, cmap='gray')
plt.show()

plt.figure()
plt.hist(image_after)
plt.show()


# plt.figure(figsize=(5, 2))
# plt.subplot(121)
# plt.imshow(image, cmap='gray')
# plt.subplot(122)
# plt.imshow(image, cmap='gray')
# plt.show()