# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 11:18:03 2022

@author: manos
"""
from metrics import dice_loss, dice_coef, iou
from train import read_image

from tensorflow.keras.utils import CustomObjectScope
from natsort import natsorted
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import cv2
import os


def compute_compliance_from_excel(patiend_id, excel_path, asc_or_desc='asc'):
    df = pd.read_excel(excel_path, sheet_name='Compliance Dijon', index_col=0)
    
    syst_press, diast_press, asc_min, asc_max, asc_compliance, \
        asc_distensibility, desc_min, desc_max, \
        desc_compliance, desc_distensibility = df.loc[patiend_id, 'PS':]
        
    if asc_or_desc == 'asc':
        compliance = compute_compliance(asc_min, asc_max, syst_press, diast_press), asc_compliance
        return compliance, asc_min, asc_max
    else:
        compliance = compute_compliance(desc_min, desc_max, syst_press, diast_press), desc_compliance
        return compliance, desc_min, desc_max
    

def fetch_compliance_from_excel(patiend_id, excel_path, asc_or_desc='asc'):
    df = pd.read_excel(excel_path, sheet_name='Compliance Dijon', index_col=0)
            
    compliance = None
    if asc_or_desc == 'asc':
        compliance = df.loc[patiend_id, 'asc-compliance']
    else:
        compliance = df.loc[patiend_id, 'desc-compliance']
        
    return compliance


def fetch_syst_press_from_excel(patiend_id, excel_path):
    df = pd.read_excel(excel_path, sheet_name='Compliance Dijon', index_col=0)
    return df.loc[patiend_id, 'PS']


def fetch_diast_press_from_excel(patiend_id, excel_path):
    df = pd.read_excel(excel_path, sheet_name='Compliance Dijon', index_col=0)
    return df.loc[patiend_id, 'PD']

        
def compute_compliance(min_area, max_area, syst_press, diast_press):
    return np.abs(max_area - min_area) / np.abs(syst_press - diast_press)


def segment_aorta_and_calculate_area(model, images, display=False):
    """
    Segment aorta and calculate the area

    Parameters
    ----------
    model : TYPE
        DESCRIPTION.
    images : TYPE
        DESCRIPTION.

    Returns
    -------
    Minimum and maximum area of aorta

    """
    areas = []
    for i, x in tqdm(enumerate(images), total=len(images)):
        x = read_image(x)
        
        """ Predicting the mask """
        y_pred = model.predict(np.expand_dims(x, axis=0))[0] > 0.5
        y_pred = y_pred.astype(np.float32)

        """ Compute the area of aorta """
        area = cv2.countNonZero(y_pred)
        areas.append(area)

        if display:
            """ Show the segmented aorta """
            plt.subplot(title='Predicted image & mask')
            plt.imshow(x, cmap='gray')
            plt.imshow(y_pred, cmap='jet', alpha=0.2)
            plt.show()
        
    """ Get the min and max area """
    min_area = min(areas)
    max_area = max(areas)
    
    return min_area, max_area


if __name__ == '__main__':
    EXPERIMENT = 'exp3'
    patient_id = 'D-0001'
    
    """ File paths """
    DATASET_FOLDER_PATH = os.path.join('..', 'dataset', 'diana_segmented', '000000436346_PATRU OLIVIER')
    excel_path = os.path.join('..', 'dataset', 'diana_segmented', 'Patient_Compliance_Dec2020.xlsx')
       
    """ Loading patient images """
    images = glob.glob(f'{DATASET_FOLDER_PATH}/**/*.IMA', recursive=True)
    
    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model(os.path.join('..', "output", EXPERIMENT, "model.h5"))
    
    """ Perform aorta segmentation """
    """ Compute areas on every slice """
    min_area, max_area = segment_aorta_and_calculate_area(model, images)
        
    """ Fetch compliance from excel file """
    original_compliance, original_min, original_max = compute_compliance_from_excel(patient_id, excel_path)
    
    """ Fetch systolic and distolic pressures from excel file """
    syst_press = fetch_syst_press_from_excel(patiend_id, excel_path)
    diast_press = fetch_diast_press_from_excel(patiend_id, excel_path)
    
    """ Compute compliances on every slice and then calculate the average, for global compliance """
    computed_compliances = []
    for image in images:        
        """ Compute compliance """
        computed_compliance = compute_compliance(min_area, max_area, syst_press, diast_press)
        computed_compliances.append(computed_compliance)
        
    """ Compute global compliance """
    avg_computed_compliance = np.mean(np.array(computed_compliances))
        
    print('Computed global ascending compliance', avg_computed_compliance)
    print('Original global ascending compliance', original_compliance)
