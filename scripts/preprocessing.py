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


def bm3d_denoising(image, display=True):
    bm3d_cleaned = bm3d.bm3d(image, sigma_psd=0.05, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    if display:
        plt.figure()
        plt.subplot(131, title='input image')
        plt.imshow(image, cmap='gray')
        plt.subplot(132, title='bm3d denoising')
        plt.imshow(bm3d_cleaned, cmap='gray')
        plt.show()
    return bm3d_cleaned


def n4_bias_field_correction(image, display=False):
    sitk_image = sitk.GetImageFromArray(image)
    sitk_mask = sitk.OtsuThreshold(sitk_image, 0, 1)
    
    casted_image = sitk.Cast(sitk_image, sitk.sitkFloat32)
    
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    
    numberFittingLevels = 3

    corrector.SetMaximumNumberOfIterations([50, 50] * numberFittingLevels)

    # corrected_image = corrector.Execute(image, maskImage)

    corrected_image = corrector.Execute(casted_image, sitk_mask)

    # log_bias_field = corrector.GetLogBiasFieldAsImage(sitk_image)

    # corrected_image_full_resolution = sitk_image / sitk.Exp( log_bias_field )

    
    output = sitk.GetArrayFromImage(corrected_image)
    if display:
        plt.figure()
        plt.subplot(131, title='input image')
        plt.imshow(image, cmap='gray')
        plt.subplot(132, title='n4bf correction')
        plt.imshow(output, cmap='gray')
        plt.show()
    
    return output

def intensity_range_standardization(images):
    irs = IntensityRangeStandardization()
    trained_model, transformed_images = irs.train_transform(images)
    return transformed_images


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


def contrast_stretching(image, display=False):
    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    
    if display:
        plt.figure()
        plt.subplot(131, title='before stretching')
        plt.imshow(image, cmap='gray')
        plt.subplot(132, title='after stretching')
        plt.imshow(img_rescale, cmap='gray')
        # plt.subplot(133, title='resizing to original')
        # plt.imshow(resizing_func(x, image.shape[0], image.shape[1]), cmap='gray')
        plt.show()
        
    return img_rescale

def equalize_histogram(image, display=True):
    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))

    # Equalization
    img_eq = exposure.equalize_hist(image)

    # Adaptive Equalization
    img_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)

    if display:
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
    
    return img_rescale
    
def upscale_with_padding(img, new_image_size=(256, 256),color=(0,0,0)):
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

    
def resize(image, W=512, H=512, display=True):
    # resizing_func = lambda im, W,H : cv2.resize(image, (W, H))
    # resizing_func = lambda: im, W,H : upscale_with_padding(image, (W,H))
    resizing_func = lambda im, W,H : skimage.transform.resize(im, (W,H), preserve_range=True, mode='constant', anti_aliasing=True) 
    
    x = resizing_func(image, W,H)
    
    if display:
        plt.figure()
        plt.subplot(131, title='before resizing')
        plt.imshow(image, cmap='gray')
        plt.subplot(132, title='after resizing')
        plt.imshow(x, cmap='gray')
        plt.subplot(133, title='resizing to original')
        plt.imshow(resizing_func(x, image.shape[0], image.shape[1]), cmap='gray')
        plt.show()

    return x


def crop_and_pad(img, cropx, cropy, display=False):
    h, w = img.shape
    starty = startx = 0
    # print('Input: ',img.shape)
    
    # Crop only if the crop size is smaller than image size
    if cropy <= h:   
        starty = h//2-(cropy//2)    
        
    if cropx <= w:
        startx = w//2-(cropx//2)
        
    cropped_img = img[starty:starty+cropy,startx:startx+cropx]
    # print('Cropped: ',cropped_img.shape)
    
    # Add padding, if the image is smaller than the desired dimensions
    old_image_height, old_image_width = cropped_img.shape
    new_image_height, new_image_width = old_image_height, old_image_width
    
    if old_image_height < cropy:
        new_image_height = cropy
    if old_image_width < cropy:
        new_image_width = cropy
    
    if (old_image_height != new_image_height) or (old_image_width != new_image_width):
    
        padded_img = np.full((new_image_height, new_image_width), 0, dtype=np.float32)
    
        x_center = (new_image_height - old_image_width) // 2
        y_center = (new_image_width - old_image_height) // 2
        
        padded_img[y_center:y_center+old_image_height, x_center:x_center+old_image_width] = cropped_img
        
        # print('Padded: ',padded_img.shape)
        result = padded_img
    else:
        result = cropped_img
        
    # print('Result: ',result.shape)
        
    if display:
        plt.figure()
        plt.subplot(131, title='before cropping')
        plt.imshow(img, cmap='gray')
        plt.subplot(132, title='after cropping')
        plt.imshow(result, cmap='gray')
        # plt.subplot(133, title='resizing to original')
        # plt.imshow(resizing_func(x, image.shape[0], image.shape[1]), cmap='gray')
        plt.show()
        
    return result


def normalize(image):
    x = image/np.max(image)
    x = x.astype(np.float32)
    
    return x


def limiting_filter(img, threshold=8, display=False):
    ret1, th1 = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    binary_mask = img > ret1
    output = np.zeros_like(img)
    output[binary_mask] = img[binary_mask]
    
    if display:
        fig, ((ax1,ax2), (ax3,ax4)) = plt.subplots(2, 2)
        ax1.imshow(img, cmap="gray")
        ax1.title.set_text("Original Image")
        ax2.imshow(img>0, cmap="gray")
        ax2.title.set_text("Pixels > 0")

        ax3.imshow(output, cmap="gray")
        ax3.title.set_text("Output")
        ax4.imshow(output>0, cmap="gray")
        ax4.title.set_text("After limiting Pixels > 0")
        plt.show()
        
    return output


if __name__ == "__main__":
    IMAGES_PATH = os.path.join('dataset', 'images')
    MASKS_PATH = os.path.join('dataset', 'masks')
    
    images = [pydicom.read_file(IMAGES_PATH + os.sep + s) for s in natsorted(os.listdir(IMAGES_PATH))]
    # masks = [cv2.imread(MASKS_PATH + os.sep + s, cv2.IMREAD_GRAYSCALE) for s in natsorted(os.listdir(MASKS_PATH))]
    
    random.shuffle(images)
    
    rand_images = []
    for j in range(20):
        rand = random.randint(0, len(images)-1)
        rand_images.append(images[rand].pixel_array)
        
    # rand_images = [n4_bias_field_correction(i) for i in rand_images]
    # rand_images = [bm3d_denoising(i) for i in rand_images]
    rand_images = [limiting_filter(i) for i in rand_images]
    rand_images = [contrast_stretching(i) for i in rand_images]
    rand_images = [normalize(i) for i in rand_images]
    rand_images = [crop_and_pad(i, 256, 256) for i in rand_images]
    # rand_images = [equalize_histogram(i) for i in rand_images]
    
    # rand_images = [resize(i.pixel_array) for i in rand_images]
    # rand_images = [standardization(i) for i in rand_images]
    # rand_images = [intensity_range_standardization(i) for i in rand_images]
    
    # image_after = images[rand]
    # mask = masks[rand]
    
    
    # plt.figure()
    # plt.subplot(121, title='before')
    # plt.imshow(image_before, cmap='gray')
    # plt.subplot(122, title='after')
    # plt.imshow(image_after, cmap='gray')
    # plt.show()
    
