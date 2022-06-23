import numpy as np
from scipy.interpolate import interpn
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from natsort import natsorted
import matplotlib.pyplot as plt
import pydicom
import cv2
import os

def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw


def signed_bwdist(im):
    '''
    Find perim and return masked image (signed/reversed)
    '''    
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im

def bwdist(im):
    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im

def interpolate(top, bottom, precision):
    '''
    Interpolate between two contours

    Input: top 
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate 
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if precision>2:
        print("Error: Precision must be between 0 and 1 (float)")

    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1+precision

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # create ndgrids 
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))
    xi = np.c_[np.full((r*c),precision), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out

def press(event):
    global the_key
    the_key = event.key

if __name__ == "__main__":
    IMAGES_PATH = '../dataset/diana_segmented/000000013461/2.16.840.1.113669.632.20.634880955.537038478.10003134441/31'#os.path.join('..', 'dataset', 'images')
    MASKS_PATH = '../results/res_unet-diana_healthy_marfan-lr_0.001-batch_8-augmented-instance_normalization-polygon2mask-Kfield/1/000000013461/masks'#os.path.join('..', 'dataset', 'masks')
    
    images = [(pydicom.read_file(IMAGES_PATH + os.sep + s)).pixel_array for s in natsorted(os.listdir(IMAGES_PATH))]
    masks = [plt.imread(MASKS_PATH + os.sep + s, cv2.IMREAD_GRAYSCALE)[:,:,0] for s in natsorted(os.listdir(MASKS_PATH))]
    
    # # Run interpolation
    # import nibabel as nib
    # path = "../data/mask"
    # imgs=[]
    # for file in sorted(os.listdir(path))[1:5]:
    #     img = nib.load(os.path.join(path,file))
    #     print(f"For file {file}, img: {img.shape}")
    #     imgs.append(img.get_fdata().transpose(2,1,0))
    
    for i, (image, mask) in enumerate(zip(images, masks)):
        if (i == 0) or (i == len(images)-1):
            continue
        
        print(f"i:{i}, image shape: {images[i].shape}, mask shape: {masks[i].shape, masks[i].dtype}")
        
        out = interpolate(masks[i-1], masks[i+1], 0.1)

        # plt.subplot(141)
        # plt.imshow(masks[i-1])
        # plt.subplot(142)
        # plt.imshow(out)
        # plt.subplot(143)
        # plt.imshow(masks[i+1])
        # plt.subplot(144)
        # plt.imshow(images[i])
        # plt.show()
        
        plt.subplot(121)
        plt.imshow(images[i], cmap='gray')
        plt.imshow(masks[i], cmap='jet', alpha=0.2)
        plt.subplot(122)
        plt.imshow(images[i], cmap='gray')
        plt.imshow(out, cmap='jet', alpha=0.2)
        plt.show()
        
        plt.gcf().canvas.mpl_connect('key_press_event', press)
        while not plt.waitforbuttonpress(): pass 
        plt.close()