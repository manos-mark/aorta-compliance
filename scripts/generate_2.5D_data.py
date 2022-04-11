# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 16:29:30 2022

@author: manos
"""

import os
import glob
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pydicom
import numpy as np
import shutil
from skimage import exposure
from tqdm import tqdm
from natsort import natsorted
import shutil
import nibabel as nib


DICOMS_PATH = os.path.join('..', 'dataset', 'images') 
MASKS_PATH = os.path.join('..', 'dataset', 'masks') 

NEW_IMAGES_PATH = os.path.join('..', 'dataset', '2.5D_images') 
NEW_MASKS_PATH = os.path.join('..', 'dataset', '2.5D_masks') 



def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)
        
        
if __name__ == '__main__':
    create_dir(NEW_IMAGES_PATH)
    create_dir(NEW_MASKS_PATH)

    image_paths = natsorted(os.listdir(DICOMS_PATH))
    mask_paths = natsorted(os.listdir(MASKS_PATH))

    patient_ids = np.unique([i.split('_')[0] for i in image_paths])
    
    for patient_id in tqdm(patient_ids):
        patient_imgs = [img_path for img_path in image_paths if patient_id in img_path]
        
        for i, img_path in enumerate(patient_imgs):
            if (i == 0) or (i+1 >= len(patient_imgs)): continue
            
            # print(os.path.join(DICOMS_PATH, patient_imgs[i-1]))
            # print(os.path.join(DICOMS_PATH, img_path))
            # print(os.path.join(DICOMS_PATH, patient_imgs[i+1]))
            
            dcm1 = pydicom.dcmread(os.path.join(DICOMS_PATH, patient_imgs[i-1]))
            img1 = dcm1.pixel_array
            dcm2 = pydicom.dcmread(os.path.join(DICOMS_PATH, img_path))
            img2 = dcm2.pixel_array
            dcm3 = pydicom.dcmread(os.path.join(DICOMS_PATH, patient_imgs[i+1]))
            img3 = dcm3.pixel_array

            new_img = np.zeros((img1.shape[0], img1.shape[1], 3))
            
            new_img[:,:,0] = img1
            new_img[:,:,1] = img2
            new_img[:,:,2] = img3
            
            
            new_img = new_img.transpose(1,0,2)
            img_name = img_path.split('.')[0] + '.nii.gz'
            nifti_img = nib.Nifti1Image(new_img, affine=np.eye(4))
            nib.save(nifti_img, os.path.join(NEW_IMAGES_PATH, img_name))
            
            # print(os.path.join(MASKS_PATH, mask_name))
            # print('\n')
            img_name = img_path.split('.')[0] + '.png'
            mask_path = os.path.join(MASKS_PATH, img_name)
            shutil.copy(mask_path, NEW_MASKS_PATH)
                
