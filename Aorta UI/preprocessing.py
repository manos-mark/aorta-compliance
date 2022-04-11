#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 12:20:36 2022

@author: manos
"""
from skimage import exposure
import numpy as np

def crop_and_pad(img, cropy, cropx, display=False):
    h, w = img.shape
    starty = startx = 0
    # print('Input: ',img.shape)
    
    # Crop only if the crop size is smaller than image size
    if cropy <= h :   
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
    if old_image_width < cropx:
        new_image_width = cropx
    
    if (old_image_height != new_image_height) or (old_image_width != new_image_width):
    
        padded_img = np.full((new_image_height, new_image_width), 0, dtype=np.float32)
    
        x_start = (new_image_width - old_image_width) // 2
        y_start = (new_image_height - old_image_height) // 2
        
        # x_center = 0 if x_center < 0 else x_center
        # y_center = 0 if y_center < 0 else y_center
        padded_img[y_start:y_start+old_image_height, x_start:x_start+old_image_width] = cropped_img
        
        # print('Padded: ',padded_img.shape)
        result = padded_img
    else:
        result = cropped_img
        
    # print('Result: ',result.shape)
    return result

def contrast_stretching(image, display=False):
    # Contrast stretching
    p2, p98 = np.percentile(image, (2, 98))
    img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
    return img_rescale
