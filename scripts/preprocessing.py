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
from scipy import ndimage
import numpy as np
import bm3d
import SimpleITK as sitk
import cv2
from medpy.filter import IntensityRangeStandardization

import skimage
from skimage.metrics import peak_signal_noise_ratio
from skimage import morphology, data, img_as_float, exposure


def standardization(image):
    # Get brain mask
    mask = image == 0
    
    selem = morphology.disk(2)
    
    segmentation = morphology.dilation(mask, selem)
    labels, label_nb = ndimage.label(segmentation)

    mask = labels == 0
    
    mask = morphology.dilation(mask, selem)
    mask = ndimage.binary_fill_holes(mask)
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

def plot_img_and_hist(image, axes, bins=256):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, 1)
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf


def equalize_histogram(image):
    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(image)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 4), dtype=np.object)
    axes[0, 0] = fig.add_subplot(2, 4, 1)
    for i in range(1, 4):
        axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
    for i in range(0, 4):
        axes[1, i] = fig.add_subplot(2, 4, 5+i)

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(image, axes[:, 0])
    ax_img.set_title('Low contrast image')

    y_min, y_max = ax_hist.get_ylim()
    ax_hist.set_ylabel('Number of pixels')
    ax_hist.set_yticks(np.linspace(0, y_max, 5))

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_rescale, axes[:, 1])
    ax_img.set_title('Contrast stretching')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_eq, axes[:, 2])
    ax_img.set_title('Histogram equalization')

    ax_img, ax_hist, ax_cdf = plot_img_and_hist(img_adapteq, axes[:, 3])
    ax_img.set_title('Adaptive equalization')

    ax_cdf.set_ylabel('Fraction of total intensity')
    ax_cdf.set_yticks(np.linspace(0, 1, 5))

    # prevent overlap of y-axis labels
    fig.tight_layout()
    plt.show()
    
def upscale_with_padding(img, new_image_size=(512, 512),color=(0,0,0)):
    """
    This function used in order to keep the geometry of the image the same during the resize method.
    """
    if len(img.shape)==2:
        # The image is grayscaled
        # print("The image is grayscaled")
        old_image_height, old_image_width = img.shape

        # try:
        # assert channels == len(color),f"The image should be RGB -> Image channels {channels}"
        result = np.full((new_image_size[0],new_image_size[1]), 1, dtype=np.uint8)
        # # compute center offset
        x_center = (new_image_size[0] - old_image_width) // 2
        y_center = (new_image_size[1] - old_image_height) // 2

        
    elif len(img.shape)==3:
        old_image_height, old_image_width, channels = img.shape

        # try:
        assert channels == len(color),f"The image should be RGB -> Image channels {channels}"
        result = np.full((new_image_size[0],new_image_size[1], channels), color, dtype=np.uint8)
        # # compute center offset
        x_center = (new_image_size[0] - old_image_width) // 2
        y_center = (new_image_size[1] - old_image_height) // 2

    # # copy img image into center of result image
    result[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = img
    return result

    
def convert_and_resize(image, W=512, H=512, display=True):
    # resizing_func = lambda im, W,H : cv2.resize(image, (W, H))
    # resizing_func = lambda: im, W,H : upscale_with_padding(image, (W,H))
    resizing_func = lambda im, W,H : skimage.transform.resize(im, (W,H), preserve_range=True, mode='constant', anti_aliasing=True) 
    
    x = resizing_func(image, W,H)
    x = x/np.max(x)
    x = x.astype(np.float32)
    
    plt.figure()
    plt.subplot(131, title='before resizing')
    plt.imshow(image, cmap='gray')
    plt.subplot(132, title='after resizing')
    plt.imshow(x, cmap='gray')
    plt.subplot(133, title='resizing to original')
    plt.imshow(resizing_func(x, image.shape[0], image.shape[1]), cmap='gray')
    plt.show()

    return x


IMAGES_PATH = os.path.join('dataset', 'images')
MASKS_PATH = os.path.join('dataset', 'masks')

images = [pydicom.read_file(IMAGES_PATH + os.sep + s) for s in natsorted(os.listdir(IMAGES_PATH))]
# masks = [cv2.imread(MASKS_PATH + os.sep + s, cv2.IMREAD_GRAYSCALE) for s in natsorted(os.listdir(MASKS_PATH))]

random.shuffle(images)

# rand = random.randint(0, len(images)-1)
# image_before = images[rand].pixel_array

images = [convert_and_resize(i.pixel_array) for i in images]
# images = [n4_bias_field_correction(i.pixel_array) for i in images]
# images = [bm3d_denoising(i) for i in images]
# images = [standardization(i) for i in images]
# images = [equalize_histogram(i) for i in images]
# images = [intensity_range_standardization(i) for i in images]

# image_after = images[rand]
# mask = masks[rand]


# plt.figure()
# plt.subplot(121, title='before')
# plt.imshow(image_before, cmap='gray')
# plt.subplot(122, title='after')
# plt.imshow(image_after, cmap='gray')
# plt.show()

