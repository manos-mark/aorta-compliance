import os
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pydicom


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

def draw_contours(img_mri, true_mask, predicted):
    """
    This function draws both the predicted mask and ground truth mask
    The red contours are the predicted
    The blue contours are the ground truth
    """
    contours_pred = getContours(predicted)
    contours_true = getContours(true_mask)
    with_pred = cv2.drawContours(img_mri, contours_pred, -1, (0,0,255), 1)
    # combined_img = cv2.drawContours(with_pred, contours_true, -1, (255,0,0), 1)
    
    return with_pred


def getContours(im):
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours


def main():    
    predicted_mask_paths = os.listdir(PRED_PATH)
    slices_count = len(predicted_mask_paths)

    pred_path = f"{PRED_PATH}/{patient_id}_{1}.png"
    ground_truth_path = f"{MASKS_PATH}/{patient_id}_{1}.png"
    img_path = f"{IMAGES_PATH}/{patient_id}_{1}.dcm"

    predicted = cv2.imread(pred_path)
    true_mask = cv2.imread(ground_truth_path)
    
    img_mri = (pydicom.dcmread(img_path)).pixel_array
    img_mri = (img_mri - np.min(img_mri)) / (np.max(img_mri) - np.min(img_mri))
    img = np.zeros((img_mri.shape[0], img_mri.shape[1], 3))
    img[:,:,0] = img[:,:,1] = img[:,:,2] = img_mri[:,:]
    
    new_img = draw_contours(img.astype('float32'), true_mask, predicted)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    axcolor = 'yellow'
    ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor=axcolor)
    slider = Slider(ax_slider, 'Images', 1, slices_count, valinit=1, valstep=1)
    ax.imshow(new_img)
    slider.on_changed(update)
    plt.show()

def update(val):
    pred_path = f"{PRED_PATH}/{patient_id}_{int(val)}.png"
    ground_truth_path = f"{MASKS_PATH}/{patient_id}_{int(val)}.png"
    img_path = f"{IMAGES_PATH}/{patient_id}_{int(val)}.dcm"
    print(int(val))

    predicted = cv2.imread(pred_path)
    true_mask = cv2.imread(ground_truth_path)

    img_mri = (pydicom.dcmread(img_path)).pixel_array
    img_mri = (img_mri - np.min(img_mri)) / (np.max(img_mri) - np.min(img_mri))
    img = np.zeros((img_mri.shape[0], img_mri.shape[1], 3))
    img[:,:,0] = img[:,:,1] = img[:,:,2] = img_mri[:,:]

    new_img = draw_contours(img.astype('float32'), true_mask, predicted)
    new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)

    ax.imshow(new_img)
    fig.canvas.draw_idle()

if __name__ == '__main__':
    EXPERIMENT = 'unet-diana-lr_0.0001-batch_8-augmented'
    IMAGES_PATH = os.path.join('..', 'dataset', 'images')
    MASKS_PATH = os.path.join('..', 'dataset', 'masks')

    patient_ids = os.listdir(os.path.join('..', 'dataset', 'diana_segmented'))
    patient_id = '000000013461'#patient_ids[0]
    PRED_PATH = os.path.join('..', 'results', EXPERIMENT, patient_id, 'masks')

    fig, ax = plt.subplots()
    main()
