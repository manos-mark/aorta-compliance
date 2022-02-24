import os
import glob
import json
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from pydicom import dcmread
import numpy as np
import shutil
from skimage import exposure
from tqdm import tqdm
from natsort import natsorted
import nibabel as nib

DATASET_FOLDER_PATH = os.path.join('..', 'dataset', 'diana_segmented') # = 'D:\\Vibot\\thesis\\dataset\\patients'
DICOMS_PATH = os.path.join('..', 'dataset', 'images-test') # = 'D:\\Vibot\\thesis\\dataset\\dicoms'
MASKS_PATH = os.path.join('..', 'dataset', 'masks-test') # = 'D:\\Vibot\\thesis\\dataset\\masks'


def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)


def generate_polygon(image_path, points, display=False):
    image = dcmread(image_path)
    polygon = Image.new("L", (image.Columns, image.Rows))

    draw = ImageDraw.Draw(polygon)
    draw.polygon(points, fill=1)

    polygon = np.array(polygon).astype(bool)
    image = image.pixel_array

    if display:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.imshow(polygon, cmap='jet', alpha=0.2)
        ax.set_title(image_path)
        plt.show()

    return polygon


def save_as_nifti(array, name, path, affine=np.eye(4)):
    res = nib.Nifti1Image(array, affine)
    nib.save(res, os.path.join(path, f'{name}.nii.gz'))


if __name__ == '__main__':
    create_dir(DICOMS_PATH)
    create_dir(MASKS_PATH)
    create_dir

    dicoms_path = natsorted(glob.iglob('../dataset/diana_segmented/**/*.IMA', recursive=True))
    masks_path = natsorted(glob.iglob('../dataset/diana_segmented/**/ComplianceAscending.json', recursive=True))

    for mask_path in tqdm(masks_path):
        # Open contours file
        with open(mask_path) as json_file:
            data = json.load(json_file)
            
            images = []
            masks = []
            dicom_name = mask_name = None
            
            # For each patient there are a lot of slides, so we need to process each slide
            for slide_id in data:
                # Access the points of each slide
                points = data[slide_id][0]['points']

                # Convert object to list of points
                points_list = []
                for point in points:
                    x = point['x']
                    y = point['y']
                    points_list.append((x, y))    
                # print(json.dumps(data, indent=4))
                
                # Import image slide
                image_path = mask_path
                image_path = image_path.replace('contours' + os.sep, '')
                image_path = image_path.split(os.sep)[:-2]
                case_id = image_path[-3]
                patient_id = image_path[-1]
                
                image_path = os.sep.join(image_path)
                image_path = os.path.join(image_path, slide_id)
                images.append((dcmread(image_path)).pixel_array)

                # Generate ROI's polygon 
                polygon = generate_polygon(image_path, points_list)
                masks.append(polygon)

                # Save ROI image
                slide_name = slide_id.split(".")[:-1]
                slide_name[-1] += '_ROI.png'
                slide_name = ".".join(slide_name)
                
                mask_name = case_id + '_' + patient_id + '_ROI'#+ '_' + slide_name
                mask_name = mask_name.replace(" ", "_")
                dicom_name = case_id + '_' + patient_id #+ '_' + slide_id
                dicom_name = dicom_name.replace(" ", "_")
#                plt.imsave(f'{os.path.join(MASKS_PATH, mask_name)}', polygon, cmap='gray')
#                shutil.copyfile(f'{image_path}', f'{os.path.join(DICOMS_PATH, dicom_name)}')
                # image.save_as(f'{os.path.join(DICOMS_PATH, dicom_name)}')
                
            if (dicom_name is not None) and (mask_name is not None):
                save_as_nifti(np.array(images).transpose([2,1,0]), dicom_name, DICOMS_PATH)
                save_as_nifti(np.array(masks, dtype=np.uint16).transpose([2,1,0]), mask_name, MASKS_PATH)
            
