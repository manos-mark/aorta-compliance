# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:18:03 2022

@author: manos
"""
from preprocessing import crop_and_pad
from metrics import dice_loss, dice_coef, iou, hausdorff
from train import read_image, read_mask
from utils import create_dir, bland_altman_plot

from tensorflow.keras.utils import CustomObjectScope
from natsort import natsorted
from pydicom import dcmread
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import pyCompare
import math
import glob
import cv2
import os


def compute_compliance_from_excel(patient_id, excel_path, asc_or_desc='asc'):
    df = pd.read_excel(excel_path, index_col=0)
    
    resolution, syst_press, diast_press, asc_min, asc_max, asc_compliance, \
        asc_distensibility, desc_min, desc_max, \
        desc_compliance, desc_distensibility = df.loc[patient_id, 'Resolution':]
        
    if asc_or_desc == 'asc':
        compliance = compute_compliance(asc_min, asc_max, syst_press, diast_press)
        return compliance, asc_min, asc_max, resolution
    else:
        compliance = compute_compliance(desc_min, desc_max, syst_press, diast_press)
        return compliance, desc_min, desc_max, resolution
    

def fetch_compliance_from_excel(patient_id, excel_path, asc_or_desc='asc'):
    df = pd.read_excel(excel_path, index_col=0)
            
    compliance = None
    if asc_or_desc == 'asc':
        compliance = df.loc[patient_id, 'asc-compliance']
    else:
        compliance = df.loc[patient_id, 'desc-compliance']
        
    return compliance


def fetch_syst_press_from_excel(patient_id, excel_path):
    df = pd.read_excel(excel_path, index_col=0)
    return df.loc[patient_id, 'PS']


def fetch_diast_press_from_excel(patient_id, excel_path):
    df = pd.read_excel(excel_path, index_col=0)
    return df.loc[patient_id, 'PD']

        
def compute_compliance(min_area, max_area, syst_press, diast_press):
    return np.abs(max_area - min_area) / np.abs(syst_press - diast_press)


def segment_aorta(model, image, display=False):
    """ Predicting the mask """
    y_pred = model.predict(np.expand_dims(image, axis=0))[0] > 0.5
    y_pred = y_pred.astype(np.float32)

    if display:
        """ Show the segmented aorta """
        plt.subplot(title='Predicted image & mask')
        plt.imshow(image, cmap='gray')
        plt.imshow(y_pred, cmap='jet', alpha=0.2)
        plt.show()

    return y_pred


if __name__ == '__main__':
    EXPERIMENT = 'unet-diana-lr_0.001-batch_8-augmented' # 'unet-diana-lr_0.001-batch_8-augmented'
    
    """ File paths """
    excel_path = os.path.join('..', 'dataset', 'Diana_Compliance_Dec2020.xlsx')
    DATASET_FOLDER_PATH = os.path.join('..', 'dataset', 'diana_segmented')
    patient_ids = os.listdir(DATASET_FOLDER_PATH)

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss, 'hausdorff': hausdorff}):
        model = tf.keras.models.load_model(os.path.join('..', "output", EXPERIMENT, "model.h5"), compile=False)    
    
    predicted_compliances, original_compliances = ([] for i in range(2))
    original_min_areas, predicted_min_areas, original_max_areas, predicted_max_areas = ([] for i in range(4))
    """ Iterate through every patient """
    for patient_id in patient_ids[4:6]:
        patient_folder_path = os.path.join(DATASET_FOLDER_PATH, patient_id)  
        patient_output_folder_path = os.path.join('..', 'results', patient_id)
        masks_output_folder_path = os.path.join('..', 'results', patient_id, 'masks')
        create_dir(patient_output_folder_path)
        create_dir(masks_output_folder_path)
    
        """ Loading patient images """
        image_paths = glob.glob(f'{patient_folder_path}/**/*.IMA', recursive=True)
     
        """ Loading patient predicted masks """
        masks = glob.glob(f'{masks_output_folder_path}/**/*.png', recursive=True)
        
        """ Fetch compliance from excel file """
        original_compliance, original_min, original_max, resolution = compute_compliance_from_excel(patient_id, excel_path)
        original_compliances.append(original_compliance)
        
        """ Fetch systolic and distolic pressures from excel file """
        syst_press = fetch_syst_press_from_excel(patient_id, excel_path)
        diast_press = fetch_diast_press_from_excel(patient_id, excel_path)
        
        """ Segment the aorta and calculate area for each slice """
        area_per_slice = []
        
        rows = int(math.ceil(len(image_paths)/5))
        fig, axs = plt.subplots(rows, 5, figsize=(30,30))

        i = j = 0
        for k, image_path in enumerate(tqdm(image_paths, total=len(image_paths))):  
            mask_name = image_path.split('/')[3] + '_' + image_path.split('/')[5]
            mask_name = mask_name + '_' + str(k) + '.png'
            
            image = read_image(image_path)
            W,H = ((dcmread(image_path)).pixel_array).shape
            
            """ Segment aorta if it's not already segmented """
            if len(masks) == 0: # TODO make this len(masks) != len(image_paths)
                                # now it will give error because we have multiple
                                # scans for each patient
                aorta = segment_aorta(model, image)
                aorta = crop_and_pad(aorta[:,:,0], W, H)
                img = np.zeros((aorta.shape[0], aorta.shape[1], 3))
                img[:,:,0] = img[:,:,1] = img[:,:,2] = aorta[:,:]
                plt.imsave(os.path.join(masks_output_folder_path, mask_name), img, cmap='gray')
            
            else:
                aorta = read_mask(os.path.join(masks_output_folder_path, mask_name))
                aorta = crop_and_pad(aorta[:,:,0], W, H)
                
            
            image = crop_and_pad(image[:,:,0], W, H)
            
            axs[i,j].imshow(image, cmap='gray')
            axs[i,j].imshow(aorta, cmap='jet', alpha=0.2)
            axs[i,j].axis('off')
            if j < 4: j+=1
            else: 
                j=0
                if i < rows: i+=1
            
            """ Calculate area """
            area = cv2.countNonZero(aorta)
    
            """ Convert pixel to milimeters """
            area = int(area * resolution * resolution)
            area_per_slice.append(area)
            fig.tight_layout()
#        plt.show()
        fig.savefig(os.path.join(patient_output_folder_path, 'predicted_ROIs.jpg' ))
        plt.clf()
        
        """ Plot area over time """
        fig = plt.figure(figsize=(15,15))
        plt.subplot(title='Area over time')
        plt.plot(area_per_slice)
        plt.xlabel('Slices')
        plt.ylabel('Area')
        plt.ylim(np.min(area_per_slice)-100, np.max(area_per_slice)+100)
#        plt.show()
        plt.savefig(os.path.join(patient_output_folder_path, 'predicted_area_over_time.jpg' ))
        plt.clf()
        
        
        """ Get the minimum and maximum areas across all slices """        
        min_area = min(area_per_slice)
        max_area = max(area_per_slice)
        
        original_min_areas.append(original_min)
        original_max_areas.append(original_max)
        predicted_min_areas.append(min_area)
        predicted_max_areas.append(max_area)
        
        print('\nOriginal min, max areas: ', original_min, original_max)
        print('Predicted min, max areas: ', min_area, max_area)
        
        # """ Get the median of 5 values close to minimum and maximum areas across all slices """
        # area_per_slice.sort()
        # min_area = np.median(np.array(area_per_slice[:5]))
        # max_area = np.median(np.array(area_per_slice[5:]))
        
        """ Compute global ascending compliance """
        predicted_compliance = compute_compliance(min_area, max_area, syst_press, diast_press)
        predicted_compliances.append(predicted_compliance)
            
        print('Original ascending compliance', original_compliance)
        print('Predicted ascending compliance', predicted_compliance)
        
        print('\n ========================================================================== \n')
    
    pyCompare.blandAltman(original_compliances, predicted_compliances)
    plt.clf()
    pyCompare.blandAltman(original_min_areas, predicted_min_areas)
    plt.clf()
    pyCompare.blandAltman(original_max_areas, predicted_max_areas)
    plt.clf()