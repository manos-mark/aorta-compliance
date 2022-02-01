import os
import glob
from pydicom import dcmread
from skimage import exposure
from tqdm import tqdm
import matplotlib.pyplot as plt

DATASET_FOLDER_PATH = 'D:\\Vibot\\thesis\\dataset\\patients'
CLAHE_PATH = 'D:\\Vibot\\thesis\\dataset\\CLAHE'

def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    # Iterate through folders and file all files and folder
    dicom_paths = glob.glob(DATASET_FOLDER_PATH + '\\**\\*.dcm', recursive=True)

    for path in tqdm(dicom_paths):
        dicom = dcmread(path)

        # Perform CLAHE
        dicom.PixelData = exposure.equalize_adapthist(dicom.pixel_array).tobytes()
        
        # plt.imshow(dicom.pixel_array, cmap='gray')
        # plt.imshow(exposure.equalize_adapthist(dicom.pixel_array), cmap='gray')
        # plt.show()

        patient_path = path.split('\\')[5:-1]
        patient_path = '\\'.join(patient_path)

        create_dir(os.path.join(CLAHE_PATH, patient_path))

        name = path.split('\\')[-1]
        dicom.save_as(f'{os.path.join(CLAHE_PATH, patient_path, name)}')
