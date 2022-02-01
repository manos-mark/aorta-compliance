import tkinter as tk
from tkinter.filedialog import askdirectory

import os
import numpy as np
import pandas as pd
import nibabel as nib
from re import findall
from pydicom import dcmread

from imager import Imager
from .autoscrollbar_widget import AutoScrollbarWidget

class BaseIOWidget(tk.Frame):
    
    def import_dicom(self): # When Open Systolic Volume in the menu is selected
        filename = askdirectory()
        self.main_window.name1 = os.path.basename(os.path.normpath(filename))
        datasets = self.load_dicom(filename) # Read the nifti file in the same
        # way as in the colab code. IMPORTANT only 3d nifti. We should add a checker and update the
        # status bar    
        self.main_window.volume1 = datasets[0]     # Store the volume
        self.main_window.imager1 = Imager(datasets) # Pass the volume to the imager function
        self.main_window.status.set("Opened DICOM files")
        self.main_window.countim.grid(row = 1, column = 0, sticky = ('n'), pady=50)
        self.main_window.vol1load = 1
        self.main_window.vol1seg = 0
        self.main_window.vbar = AutoScrollbarWidget(self.main_window, orient='vertical')
        self.main_window.hbar = AutoScrollbarWidget(self.main_window, orient='horizontal')
        self.main_window.vbar.grid(row=0, column=0, sticky='nes')
        self.main_window.hbar.grid(row=0, column=0, sticky='ews')
        self.main_window.canvas1 = tk.Canvas(self.main_window, highlightthickness=0, xscrollcommand=self.main_window.hbar.set, yscrollcommand=self.main_window.vbar.set, width=600)
        self.main_window.canvas1.grid(row=0, column=0, sticky='nswe')
        self.main_window.vbar.configure(command=self.main_window.canvas1.yview)  # bind scrollbars to the canvas
        self.main_window.hbar.configure(command=self.main_window.canvas1.xview)

        self.main_window.canvas1.bind('<ButtonPress-1>', self.move_from)
        self.main_window.canvas1.bind('<B1-Motion>',     self.move_to)
        self.main_window.canvas1.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.main_window.canvas1.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.main_window.canvas1.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up

        self.main_window.imscale = 1.0
        self.main_window.imageid = None
        self.main_window.delta = 0.75        
        
        self.main_window.sn_bt.grid(row = 1, column = 0, sticky = None)
        self.main_window.prev.grid(row=1, column = 0, sticky = ('nw'), pady=50)
        self.main_window.next.grid(row=1, column=0, sticky=('ne'), pady=50)
        #self.text = self.canvas1.create_text(0, 0, anchor='nw', text=' ')
        self.main_window.show_image1(self.main_window.imager1.get_current_image())
        self.main_window.canvas1.configure(scrollregion=self.main_window.canvas1.bbox('all'))
        self.main_window.slider = tk.Scale(self.main_window, from_=1, to=self.main_window.imager1.get_num_im(), command=self.update_ind, orient='horizontal')
        self.main_window.slider.grid(row=1, column=0, sticky='ewn')
        
    def update_ind(self, event):
        self.main_window.imager1.index = self.main_window.slider.get()-1 # In any other case, hange the index accordingly
        self.main_window.show_image1(self.main_window.imager1.get_current_image())

    def load_dicom(self, folder):
        for root,_, files in (os.walk(folder)):
            shape = dcmread(os.path.join(root,files[0])).pixel_array.shape
            frames = len(os.listdir(folder))
            vol_or = np.zeros(shape=(frames,)+shape+(3,))
            files.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                        for x in findall(r'[^0-9]|[0-9]+', var)])
        for c,file1 in enumerate(files):
            if c == 0:
                ds = dcmread(os.path.join(root,file1))
            im = dcmread(os.path.join(root,file1)).pixel_array
            im = (im-im.min()) / (im.max() - im.min())
            vol_or[c,:,:,0]=vol_or[c,:,:,1]=vol_or[c,:,:,2] = im
        return vol_or, ds
            
    def save_segmentations(self): # When Save Segmentations in the menu is selected
        if np.sum(self.main_window.imager1._segmentation)==0: # If no segmentation exists in Volume 1
            self.main_window.status.set("Retrieve the segmentations first.")
        else:
            filename = askdirectory()
            self.main_window.status.set("Saving segmentations...")
            image = nib.Nifti1Image(self.imager1.get_segmentation().T, np.eye(4))
            name = self.main_window.name1 + '_seg.nii.gz'
            #image.to_filename(os.path.join(filename, name))
            nib.save(image,os.path.join(filename, name))
            self.main_window.status.set("Segmentations saved")
            f = open(os.path.join(filename, self.main_window.name1 + '_results.txt'), "w")
            f.write(self.result.get())
            f.close()
            df = pd.DataFrame(np.concatenate((self.main_window.imager1.areaquart, np.reshape(self.main_window.imager1.area, self.main_window.imager1.area.shape+(1,))),axis=1), columns=['Posterior', 'Lateral', 'Anterior', 'Medial', 'Global'])
            df.to_csv(os.path.join(filename, self.main_window.name1 + '_compliance_curves.csv'), index=False, header=True)
            
    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.main_window.canvas1.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.main_window.canvas1.scan_dragto(event.x, event.y, gain=1)

    def wheel(self, event):
        ''' Zoom with mouse wheel '''
        scale = 1.0
        # Respond to MacOS (event.num) or Windows (event.delta) wheel event
        if event.delta == 1 or event.delta == -120:
            scale        *= self.delta
            self.imscale *= self.delta
        if event.delta == -1 or event.delta == 120:
            scale        /= self.delta
            self.imscale /= self.delta
        # Rescale all canvas objects
        x = self.main_window.canvas1.canvasx(event.x)
        y = self.main_window.canvas1.canvasy(event.y)
        self.main_window.canvas1.scale('all', x, y, scale, scale)
        self.main_window.show_image1(self.main_window.imager1.get_current_image())
        self.main_window.canvas1.configure(scrollregion=self.main_window.canvas1.bbox('all'))
        
    def on_exit(self):  # Close the UI
        self.main_window.destroy()
        self.quit()
            