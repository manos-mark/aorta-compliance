import numpy as np
import pydicom as dicom
from skimage.morphology.convex_hull import convex_hull_image
from skimage.segmentation import find_boundaries
from math import pi

# Class to manage the Nifti Volume and Segmentation

class Imager:
    def __init__(self, datasets):
        self.datasets = datasets # Volume already loaded
        self._index = 0 # Index of the slice to show
        self.size = datasets[0].shape # Get size of the volume [#slices, W, H, 3]
        # Load pixel data
        self.values = datasets[0] # Get only the volume data
        # out of all the nifti file this is actually an array
        self._segmentation = np.zeros(self.size[0:3]) # Create an empty segmentation
        self._contours = 0  # To check wether to show contours or mask
        self._quarters = 0
        self._cog = 0
        # Get Voxel Size for calculation of the EF (not needed but in case Volumes want to be retrieved, Real Magnitude are computed)
        self.spacing = datasets[1].PixelSpacing
        self.area = np.zeros(shape=(self.size[0])) # Attribute to store volume of the segmentation
        self.areaquart = np.zeros(shape=(self.size[0],4))
        self.center = np.zeros(shape=(self.size[0],2))
        
    def segmentation(self, volume, ch=0, cog=0):
        if cog == 1:
            x0 = np.zeros(self.size[0])
            y0 = np.zeros(self.size[0])
            for i in range(self.size[0]):
                y_coords, x_coords = np.where((volume[i,:,:]==1))
                y0[i], x0[i] = np.mean(y_coords), np.mean(x_coords)
            y0, x0 = np.mean(y0), np.mean(x0)
            for i in range(self.size[0]):
                self.center[i,:] = y0, x0
        else:
            for i in range(self.size[0]):
                y_coords, x_coords = np.where((volume[i,:,:]==1))
                self.center[i,:]= np.mean(y_coords), np.mean(x_coords)
        if ch==1:
            for i in range(volume.shape[0]):
                volume[i,:,:] = convex_hull_image(volume[i,:,:])
        else:
            volume = volume     # Change the Seentation and calculate its Volume
        self._segmentation = volume
        for i in range(self.size[0]):
            #self.area[i] = np.sum(convex_hull_image(self._segmentation[i,:,:]==1))*self.spacing[0]*self.spacing[1]
            self.area[i] = np.sum(self._segmentation[i,:,:]==1)*self.spacing[0]*self.spacing[1]
        for i in range(self.size[0]):
            y0, x0 = self.center[i,:]
            y_coords, x_coords = np.where((self._segmentation[i,:,:]==1))
            coords = np.zeros((len(x_coords),2))
            coords[:,0] = x_coords
            coords[:,1] = y_coords
            q1 = np.array([coords[i,:] for i in range(len(coords)) if pi/4<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<3/4*pi]).astype(int)
            q2 = np.array([coords[i,:] for i in range(len(coords)) if 3/4*pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<pi]).astype(int)
            q2 = np.concatenate((q2, np.array([coords[i,:] for i in range(len(coords)) if -pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<-3/4*pi]).astype(int)))                  
            q3 = np.array([coords[i,:] for i in range(len(coords)) if -3/4*pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<-pi/4]).astype(int)
            q4 = np.array([coords[i,:] for i in range(len(coords)) if -pi/4<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<0]).astype(int)
            q4 = np.concatenate((q4, np.array([coords[i,:] for i in range(len(coords)) if 0<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<pi/4]).astype(int)))
            self.areaquart[i,0] = len(q1)*self.spacing[0]*self.spacing[1]
            self.areaquart[i,1] = len(q2)*self.spacing[0]*self.spacing[1]
            self.areaquart[i,2] = len(q3)*self.spacing[0]*self.spacing[1]
            self.areaquart[i,3] = len(q4)*self.spacing[0]*self.spacing[1]
        return volume

    
    def get_segmentation(self):     # Retrieve the segmentation
        return self._segmentation

    def del_curr_seg(self):
        self._segmentation[self._index,:,:] = self._segmentation[self._index,:,:]*0
    
    def update_seg(self, array):
        self._segmentation[self._index,:,:] = array
        self.segmentation(self._segmentation)
        
    def contours(self, value):  # Cahnge contours option
        self._contours = value

    def quarters(self, value):
        self._quarters = value

    def quarters(self, value):
        self._quarters = value
        
    @property
    def index(self):    # Get index of the slice
        return self._index
    
    def get_num_im(self):   # Get number of slices
        return self.size[0] # Num of slices is the 3rd item in size
    
    @index.setter
    def index(self, value): # Always remain in the range [0, num of slices]
        while value < 0:
            value += self.size[2]
        self._index = value % self.size[2]

    def get_image(self, index):     # Get np array of the corresponding slice
        img = self.values[index, :, :, :] # Get the slice of interest
        img = 255.0 * img # Normalize the values
        # to be plotted properly on the canvas (uint8)
        res = img.astype('uint8') # Create RGB image
        if (np.sum(self._segmentation)>0):  # If segmentation, plot it in red (mask or contours)
            if (self._quarters == 0):
                if(self._contours==0):
                    res[:,:,0] = np.where((self._segmentation[index,:,:]==1), 255, res[:,:,0])
                else:
                    res[:,:,0] = np.where(find_boundaries((self._segmentation[index,:,:]==1), mode='inner'), 255, res[:,:,0])
            else:
                if(self._contours==0):
                    y_coords, x_coords = np.where((self._segmentation[index,:,:]==1))
                    y0, x0 = self.center[index,:]
                    coords = np.zeros((len(x_coords),2))
                    coords[:,0] = x_coords
                    coords[:,1] = y_coords
                    # q1 = np.array([coords[i,:] for i in range(len(coords)) if ((coords[i,0]>x_0)&(coords[i,1]>y_0))]).astype(int)
                    # q2 = np.array([coords[i,:] for i in range(len(coords)) if ((coords[i,0]>x_0)&(coords[i,1]<y_0))]).astype(int)
                    # q3 = np.array([coords[i,:] for i in range(len(coords)) if ((coords[i,0]<x_0)&(coords[i,1]<y_0))]).astype(int)
                    # q4 = np.array([coords[i,:] for i in range(len(coords)) if ((coords[i,0]<x_0)&(coords[i,1]>y_0))]).astype(int)
                    q1 = np.array([coords[i,:] for i in range(len(coords)) if pi/4<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<3/4*pi]).astype(int)
                    q2 = np.array([coords[i,:] for i in range(len(coords)) if 3/4*pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<pi]).astype(int)
                    q2 = np.concatenate((q2, np.array([coords[i,:] for i in range(len(coords)) if -pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<-3/4*pi]).astype(int)))                  
                    q3 = np.array([coords[i,:] for i in range(len(coords)) if -3/4*pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<-pi/4]).astype(int)
                    q4 = np.array([coords[i,:] for i in range(len(coords)) if -pi/4<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<0]).astype(int)
                    q4 = np.concatenate((q4, np.array([coords[i,:] for i in range(len(coords)) if 0<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<pi/4]).astype(int)))
                    res[q1[:,1],q1[:,0],0] = 255
                    res[q2[:,1],q2[:,0],1] = 255
                    res[q3[:,1],q3[:,0],2] = 255
                    res[q4[:,1],q4[:,0],0:2] = 255
                else:
                    y_coords, x_coords = np.where(find_boundaries(self._segmentation[index,:,:]==1, mode = 'inner'))
                    y0, x0 = np.mean(y_coords), np.mean(x_coords)
                    coords = np.zeros((len(x_coords),2))
                    coords[:,0] = x_coords
                    coords[:,1] = y_coords
                    # q1 = np.array([coords[i,:] for i in range(len(coords)) if ((coords[i,0]>x_0)&(coords[i,1]>y_0))]).astype(int)
                    # q2 = np.array([coords[i,:] for i in range(len(coords)) if ((coords[i,0]>x_0)&(coords[i,1]<y_0))]).astype(int)
                    # q3 = np.array([coords[i,:] for i in range(len(coords)) if ((coords[i,0]<x_0)&(coords[i,1]<y_0))]).astype(int)
                    # q4 = np.array([coords[i,:] for i in range(len(coords)) if ((coords[i,0]<x_0)&(coords[i,1]>y_0))]).astype(int)
                    q1 = np.array([coords[i,:] for i in range(len(coords)) if pi/4<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<3/4*pi]).astype(int)
                    q2 = np.array([coords[i,:] for i in range(len(coords)) if 3/4*pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<pi]).astype(int)
                    q2 = np.concatenate((q2, np.array([coords[i,:] for i in range(len(coords)) if -pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<-3/4*pi]).astype(int)))                  
                    q3 = np.array([coords[i,:] for i in range(len(coords)) if -3/4*pi<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<-pi/4]).astype(int)
                    q4 = np.array([coords[i,:] for i in range(len(coords)) if -pi/4<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<0]).astype(int)
                    q4 = np.concatenate((q4, np.array([coords[i,:] for i in range(len(coords)) if 0<np.arctan2(coords[i,1]-y0, coords[i,0]-x0)<pi/4]).astype(int)))
                    res[q1[:,1],q1[:,0],0] = 255
                    res[q2[:,1],q2[:,0],1] = 255
                    res[q3[:,1],q3[:,0],2] = 255
                    res[q4[:,1],q4[:,0],0:2] = 255
        return res

    def get_current_image(self):    # Get current image
        return self.get_image(self.index)