# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:18:03 2022

@author: manos
"""
from preprocessing import crop_and_pad
from metrics import dice_loss, dice_coef, iou
from train import read_image

from tensorflow.keras.utils import CustomObjectScope
from natsort import natsorted
from pydicom import dcmread
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import cv2
import os


def compute_compliance_from_excel(patient_id, excel_path, asc_or_desc='asc'):
    df = pd.read_excel(excel_path, index_col=0)
    
    syst_press, diast_press, asc_min, asc_max, asc_compliance, \
        asc_distensibility, desc_min, desc_max, \
        desc_compliance, desc_distensibility = df.loc[patient_id, 'PS':]
        
    if asc_or_desc == 'asc':
        compliance = compute_compliance(asc_min, asc_max, syst_press, diast_press)
        return compliance, asc_min, asc_max
    else:
        compliance = compute_compliance(desc_min, desc_max, syst_press, diast_press)
        return compliance, desc_min, desc_max
    

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
    EXPERIMENT = 'u-net_lr_0.0001-batch_8-dice_loss-more-pretrained-augmented-multi-centre'
    patient_id = 'D-0008'
    
    """ File paths """
    DATASET_FOLDER_PATH = os.path.join('..', 'dataset', 'diana_segmented', '000000359340_BOIVIN FRANCK')
    excel_path = os.path.join('..', 'dataset', 'diana_segmented', 'Patient_Compliance_Dec2020.xlsx')
       
    """ Loading patient images """
    image_paths = glob.glob(f'{DATASET_FOLDER_PATH}/**/*.IMA', recursive=True)
    
    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(os.path.join('..', "output", EXPERIMENT, "model.h5"))    
        
    """ Fetch compliance from excel file """
    original_compliance, original_min, original_max = compute_compliance_from_excel(patient_id, excel_path)
    
    """ Fetch systolic and distolic pressures from excel file """
    syst_press = fetch_syst_press_from_excel(patient_id, excel_path)
    diast_press = fetch_diast_press_from_excel(patient_id, excel_path)
    
    """ Segment the aorta and calculate area for each slice """
    area_per_slice = []
    for image_path in tqdm(image_paths, total=len(image_paths)):  
        image = read_image(image_path)
        W,H = ((dcmread(image_path)).pixel_array).shape
        
        """ Segment aorta """
        aorta = segment_aorta(model, image)
        aorta = crop_and_pad(aorta[:,:,0], W, H)
        image = crop_and_pad(image[:,:,0], W, H)
        
        plt.subplot(title='Predicted image & mask')
        plt.imshow(image, cmap='gray')
        plt.imshow(aorta, cmap='jet', alpha=0.2)
        plt.show()
        
        """ Calculate area """
        area = cv2.countNonZero(aorta)
        area_per_slice.append(area)

    """ Plot area over time """
    plt.subplot(title='Area over time')
    plt.plot(area_per_slice)
    plt.xlabel('Slices')
    plt.ylabel('Area')
    plt.ylim(np.min(area_per_slice)-500, np.max(area_per_slice)+500)
    plt.show()
    
    
    """ Get the minimum and maximum areas across all slices """        
    min_area = min(area_per_slice)
    max_area = max(area_per_slice)
    
    print('Original min, max: ', original_min, original_max)
    print('Predicted asc_min, asc_max: ', min_area, max_area)
    
    # """ Get the median of 5 values close to minimum and maximum areas across all slices """
    # area_per_slice.sort()
    # min_area = np.median(np.array(area_per_slice[:5]))
    # max_area = np.median(np.array(area_per_slice[5:]))
    
    """ Compute global ascending compliance """
    computed_compliance = compute_compliance(min_area, max_area, syst_press, diast_press)
        
    print('Original ascending compliance', original_compliance)
    print('Predicted ascending compliance', computed_compliance)
