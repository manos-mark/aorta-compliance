# -*- coding: utf-8 -*-
"""
Created on Tue May 17 22:05:00 2022

@author: manos
"""

import os
import pydicom
from natsort import natsorted
from glob import glob
import matplotlib.pyplot as plt
from scipy import ndimage
import numpy as np
import cv2
from preprocessing import normalize, crop_and_pad
from tqdm import tqdm

def add_gaussian_noise(image, mask, image_path, mask_path, display=False):
    waves_image_path = os.path.join('..', 'dataset', 'waves.jpg')
    waves_image = cv2.imread(waves_image_path, cv2.IMREAD_GRAYSCALE)
    waves_image = normalize(waves_image)
    
    row, col = image.shape
    waves_image = cv2.resize(waves_image, dsize=(col//2,row//2), interpolation=cv2.INTER_CUBIC)
    waves_image = np.concatenate((waves_image, waves_image), axis=0)
    waves_image = np.concatenate((waves_image, waves_image), axis=1)
    waves_image = crop_and_pad(waves_image, row, col)
    
    mu, noise_ratios = 0, [0, 0.03, 0.09, 0.11] 
    waves_ratios = [0, 0.2, 0.4, 0.6]
    processed_imgs = []
    
    for i, s in enumerate(noise_ratios):
        noisy_image = image + np.random.normal(mu, s, (row,col)) 
        noisy_img_clipped = np.clip(noisy_image, 0, 255)  # we might get out of bounds due to noise
        
        temp_row = []
        for j, a in enumerate(waves_ratios):
            processed_image = (noisy_img_clipped * 1-a) + (waves_image * a)
            np.save(os.path.join('..', 'dataset', 'images', image_path+f'-Gn{s}-W{a}'), image)
            np.save(os.path.join('..', 'dataset', 'masks', mask_path+f'-Gn{s}-W{a}'), mask)
            # print(os.path.join('..', 'dataset', 'images', f'Gn_{s}-W_{a}-'+image_path))
            # print(os.path.join('..', 'dataset', 'masks', f'Gn_{s}-W_{a}-'+mask_path))
            temp_row.append(processed_image)
            
        processed_imgs.append(temp_row)
            
        
    if display:
        f, subplots = plt.subplots(4, 4)
        for i, s in enumerate(noise_ratios):
            for j, a in enumerate(waves_ratios):
                subplots[i,j].imshow(processed_imgs[i][j], cmap='gray')
                subplots[i,j].set_title(f'Gn:{s}, W:{a}', fontsize=8)
                subplots[i,j].axis('off')
            # plt.tight_layout()
            plt.show()  
          
    return processed_imgs



if __name__ == "__main__":
    IMAGES_PATH = natsorted(os.listdir(os.path.join('..', 'dataset', 'raw_images')))
    MASKS_PATH = natsorted(os.listdir(os.path.join('..', 'dataset', 'raw_masks')))
    
    for image_path, mask_path in tqdm(zip(IMAGES_PATH, MASKS_PATH), total=len(IMAGES_PATH)):
        image = (pydicom.read_file(os.path.join('..', 'dataset', 'raw_images') + os.sep + image_path)).pixel_array
        mask = cv2.imread(os.path.join('..', 'dataset', 'raw_masks') + os.sep + mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = normalize(image)
        add_gaussian_noise(image, mask, image_path, mask_path)
        
        
                 
