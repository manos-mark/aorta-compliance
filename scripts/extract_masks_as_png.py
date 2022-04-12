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
from skimage.draw import polygon2mask


# DATASET_FOLDER_PATH = os.path.join('..', 'dataset', 'diana_segmented') 
# DATASET_FOLDER_PATH = os.path.join('..', 'dataset', 'healthy_segmented') 
DATASET_FOLDER_PATH = os.path.join('..', 'dataset', 'marfan_segmented') 
DICOMS_PATH = os.path.join('..', 'dataset', 'images') 
MASKS_PATH = os.path.join('..', 'dataset', 'masks') 


def create_dir(path):
    """ Create a directory. """
    if not os.path.exists(path):
        os.makedirs(path)

def press(event):
    global the_key
    the_key = event.key

def generate_hull(image_path, points, display=True):
    image = dcmread(image_path)
    polygon = Image.new("L", (image.Columns, image.Rows))

    draw = ImageDraw.Draw(polygon)
    draw.polygon(points, fill=1)
    
    
    polygon = np.array(polygon).astype(bool)
    image = image.pixel_array
    points = np.array([list(elem) for elem in points])
    
    points_rev = np.array([[point[1], point[0]] for point in points])
    
    
    from scipy.spatial import ConvexHull
    from skimage.draw import polygon2mask
    
    pol = polygon2mask(image.shape, points_rev)
    # hull = ConvexHull(points)
    
    if display:
        f, (ax1, ax2) = plt.subplots(1,2,figsize=(15,15), sharex=True, sharey=True)
        
        # plt.imshow(image, cmap='gray')
        # plt.plot(points[:,0], points[:,1], 'o')
        # ax3.plot(points[:,0], points[:,1], 'o')
        # ax3.plot(points[hull.vertices,0], points[hull.vertices,1])
        
        ax1.imshow(image, cmap='gray')
        ax1.imshow(pol, cmap='jet',alpha=0.2)
        ax1.set_title('sk-polygon2mask')
        
        ax2.imshow(image, cmap='gray')
        ax2.imshow(polygon, cmap='jet', alpha=0.2)
        ax2.set_title('initial method')
        
        plt.draw()
        plt.tight_layout()
        plt.show()        
        plt.gcf().canvas.mpl_connect('key_press_event', press)
        while not plt.waitforbuttonpress(): pass  # ignore mouse events use by zomming ...
        plt.close(f)

    return polygon
    

def generate_polygon(image_path, points, display=False):
    image = dcmread(image_path)
    polygon = Image.new("L", (image.Columns, image.Rows))
    image = image.pixel_array

    draw = ImageDraw.Draw(polygon)
    draw.polygon(points, fill=1)
    polygon = np.array(polygon).astype(bool)

    # points = np.array([[point[1], point[0]] for point in points])  
    # polygon = polygon2mask(image.shape, points)

    if display:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.imshow(polygon, cmap='jet', alpha=0.2)
        ax.set_title(image_path)
        plt.show()

    return polygon


if __name__ == '__main__':
    create_dir(DICOMS_PATH)
    create_dir(MASKS_PATH)

    dicoms_path = natsorted(glob.iglob(f'{DATASET_FOLDER_PATH}/**/*.ima', recursive=True))
    masks_path = natsorted(glob.iglob(f'{DATASET_FOLDER_PATH}/**/ComplianceAscending.json', recursive=True))

    for mask_path in tqdm(masks_path):
        # Open contours file
        with open(mask_path) as json_file:
            data = json.load(json_file)
            
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
                scan_id = image_path[-1]
            
                image_path = os.sep.join(image_path)
                image_path = os.path.join(image_path, slide_id)

                # Generate ROI's polygon 
                try:
                    polygon = generate_polygon(image_path, points_list)
                except:
                    continue

                # Remove unnecessary info
                try:
                    slide_id = slide_id.split(".")[4]
                    slide_id = "".join(slide_id)
                except:
                    slide_id = slide_id.split(".")[0]
                    slide_id = slide_id[-3:]
                    slide_id = "".join(slide_id)

                # Strip leading zeros
                slide_id = slide_id.lstrip("0")

                # Save ROIs and images
                mask_name = case_id + '_' + slide_id + '.png'
                dicom_name = case_id + '_' + slide_id + '.dcm'

                plt.imsave(f'{os.path.join(MASKS_PATH, mask_name)}', polygon, cmap='gray')
                shutil.copyfile(f'{image_path}', f'{os.path.join(DICOMS_PATH, dicom_name)}')
