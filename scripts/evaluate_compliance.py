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
    df = pd.read_excel(excel_path, sheet_name='Compliance Dijon', index_col=0)
    
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
    df = pd.read_excel(excel_path, sheet_name='Compliance Dijon', index_col=0)
            
    compliance = None
    if asc_or_desc == 'asc':
        compliance = df.loc[patient_id, 'asc-compliance']
    else:
        compliance = df.loc[patient_id, 'desc-compliance']
        
    return compliance


def fetch_syst_press_from_excel(patient_id, excel_path):
    df = pd.read_excel(excel_path, sheet_name='Compliance Dijon', index_col=0)
    return df.loc[patient_id, 'PS']


def fetch_diast_press_from_excel(patient_id, excel_path):
    df = pd.read_excel(excel_path, sheet_name='Compliance Dijon', index_col=0)
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
    EXPERIMENT = 'exp3'
    patient_id = 'D-0002'
    
    """ File paths """
    DATASET_FOLDER_PATH = os.path.join('..', 'dataset', 'diana_segmented', '000151041932_ALEXANDRE ROBERT')
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
        aorta = crop_and_pad(aorta[:,:,0], W, H, display=True)
        
        """ Calculate area """
        area = cv2.countNonZero(aorta)
        area_per_slice.append(area)
        
    """ Compute global ascending compliance """
    min_area = min(area_per_slice)
    max_area = max(area_per_slice)
    computed_compliance = compute_compliance(min_area, max_area, syst_press, diast_press)
        
    print('Computed ascending compliance', computed_compliance)
    print('Original ascending compliance', original_compliance)
