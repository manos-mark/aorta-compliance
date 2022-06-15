from tkinter.filedialog import askdirectory
from tkinter import messagebox as mbox
import pydicom as dicom
import PIL.Image, PIL.ImageTk, os, PIL.ImageDraw
from imager import Imager
import numpy as np
import pandas as pd
from segmentation_network import SegmentationNetwork
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt 
import nibabel as nib
from tkinter import *
# from tkinter import Tk, Menu, BooleanVar, StringVar
from tkinter.ttk import *
import re
import copy
import cv2
from interpolation_functions import interpolate

class StatusBar(Frame):
    height = 19

    def __init__(self, master): # Initialize the status bar and place it in the GUI
        Frame.__init__(self, master)
        self.label = Label(self, relief='groove', anchor='w')
        self.label.pack(fill='both')

    def set(self, format_str, *args): # Change the text on the status bar
        self.label.config(text=format_str % args, font=("TkDefaultFont",12))
        self.label.update_idletasks()

    def clear(self): # Clear the text
        self.label.config(text="")
        self.label.update_idletasks()


class AutoScrollbar(Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise TclError('Cannot use place with this widget')

# Class to define structure, connections and widgets inside the GUI
class MainWindow(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent # This is the GUI object (root)
        
        # Systolic Volume will be subscripted with index 1
        self.image = None   # Define all the objects the GUI will contain
        self.photo = None   # To get the image that will be displayed using canvas
        self.canvas = None  # Object to display images
        self.imager = None  # Function to load the nifti file        
        
        # Diastolic Volume will be subscripted with index 2
        self.status = None  # Object to show some text at the bottom
        self.slice_cnt = None
        self.prev_btn = None
        self.next_btn = None
        self.volume = np.array([])   # To store both volumes
        self.filename = ''     # To store file names for optional saving
        self.init_ui()

    def init_ui(self):
        self.parent.title("Aortic Segmentation and Compliance Tool")   # Title of the GUI
        
        #self.parent.iconbitmap(resource_path('bhicon.ico'))
        self.init_toolbar() # initialize the top menu
        # For .grid Widget management
        self.parent.columnconfigure([0,1], weight=3)
        self.parent.rowconfigure([0], weight=3)
        self.parent.rowconfigure([1], weight=1)
        # Image canvas 
        self.show_contours_status = BooleanVar()
        self.split_quarters_status = BooleanVar()
        self.convex_hull_status = BooleanVar()
        self.correct_cog_status = BooleanVar()
        self.compliance = np.zeros(shape=(5,))
        self.strain = np.zeros(shape=(4,))
        self.systolic_pressure = StringVar(value='130')
        self.diastolic_pressure = StringVar(value='80')
        self.result = StringVar()
        # Load the model and initiate it (avoid of doing it when aplying the model)
        self.model = SegmentationNetwork()
        # Message Function
#        mbox.showinfo("Usage Information", ("This program is used for segmentation of the ascending aorta and calculation "
#"of the Aortic Compliance. The segmentation is based on UNet Convolutional Neural Network." 
#"\n\nThe user should provide the application with a cardiac MRI sequence in DICOM format  by going to "
#"File -> Open DICOM Sequence.\n\nThe program will showcase the images provided. Better visualization can be achieved by zooming in/out and panning."
#" By clicking on the button \"Segment Aorta\", the "
#"segmentation will be shown. The user can manually segment one frame, once the aorta is correctly zoomed, by clicking on the \"Segment manually\" "
#"button. The segmentation contour should be defined by left-clicking on the aorta contour. Specific points can be removed by selecting them on the list "
#"and clicking on \"Delete selected\". Once all the contour points are defined, the user should click on \"End segmentation\". The segmentation will be stored"
#" and the compliance recomputed."
#"\n\nSystolic and Diastolic pressures can be entered by the user (by default are fixed to 130 and 80 mmHg, "
#"respectively).\n\nBy clicking on the button \"Compute Compliance\", the global compliance together with the local compliances "
#"(if \"Quarters\" is selected, will be shown"
#" together with area of the segmented aorta over time.\nSegmentations together with the compliance results can "
#"be saved in File -> Save Segmentation."))


        self.status = StatusBar(self.parent)
        self.status.grid(row=1, column=0, sticky='SW')
        self.slice_cnt = StatusBar(self.parent)
        self.prev_btn = Button(self.parent,text = '< Previous', command = self.previous_action)
        self.next_btn = Button(self.parent,text = 'Next >', command = self.next_action)
        
        self.manual_segmentation_btn = Button(self.parent,text = 'Segment Manually', command = self.manual_segmentation_action)
        self.manual_segmentation_end_btn = Button(self.parent,text = 'End Segmentation', command = self.manual_segmentation_end_action)
        self.systolic_pressure_label = Label(self.parent, text="Systolic Pressure: ")
        self.diastolic_pressure_label = Label(self.parent, text="Diastolic Pressure: ")
        self.result_label = Label(self.parent, textvariable=self.result, relief=('raised'))
        self.systolic_pressure_entry = Entry(self.parent, textvariable=self.systolic_pressure, width=2)
        self.diastolic_pressure_entry = Entry(self.parent, textvariable=self.diastolic_pressure, width=2)
        self.segment_btn = Button(self.parent, text = 'Segment Aorta', command = self.segment_action)   
        
        red_btn_style = Style()
        red_btn_style.configure('BR.TButton', background='red', foreground='red')
        self.delete_slice_btn = Button(self.parent, text = 'Delete slice', command = self.remove_slice_segmentation, style='BR.TButton')
        
        green_btn_style = Style()
        green_btn_style.configure('BG.TButton', background='green', foreground='green')
        self.compute_compliance_btn = Button(self.parent,text = 'Compute Compliance', command = self.compute_compliance_action, style='BG.TButton')
        
        self.show_contours_checkbtn = Checkbutton(self.parent, text="Show contour", variable=self.show_contours_status, command = self.show_contours_action)
        self.convex_hull_checkbtn = Checkbutton(self.parent, text="Apply Convex Hull", variable=self.convex_hull_status, command=self.convex_hull_action)
        self.cog_checkbtn = Checkbutton(self.parent, text="Correct c.o.g.", variable=self.correct_cog_status, command=self.cog_action)
        self.split_quarters_checkbtn = Checkbutton(self.parent, text="Quarters", variable=self.split_quarters_status, command = self.split_quarters_action)
        
        # self.import_btn = Button(self.parent, text="Import Dicom", command=self.on_load_dicom, background='grey', foreground='black', height='1', font=("Helvetica 16 bold"))
        self.import_btn = Button(self.parent, text="Import Dicom", command=self.on_load_dicom)
        self.import_btn.grid(row=0, column=0, sticky='new')
#        
#        save_segmentations_btn = Button(self.parent, text = "Save Segmentations", command=self.save_segmentation, background='grey', foreground='black', height='1', font=("Helvetica 16 bold"))
#        save_segmentations_btn.grid(row=0, column=1, sticky='nsew')
#        
        # self.exit_btn = Button(self.parent, text = "Exit", command=self.on_exit, background='grey', foreground='black', height='1', font=("Helvetica 16 bold"))
        self.exit_btn = Button(self.parent, text = "Exit", command=self.on_exit)
        self.exit_btn.grid(row=0, column=1, sticky='new')
        
    def remove_slice_segmentation(self):
#        self.split_quarters_status.set(False)
#        self.imager.quarters(0)
#        self.convex_hull_status.set(False)

        self.imager.del_curr_seg()
        
        self.slider = Scale(self.parent, from_=1, to=self.imager.get_slice_cnt(), command=self.update_index, orient=HORIZONTAL)
        self.slider.grid(row=1, column=0, sticky='ewn')
        
        try:
            self.previous_action()
        except:
            self.next_action()
            
        self.slice_cnt.set(str(int(self.imager.index)+1)+"/"+str(self.imager.get_slice_cnt()))
        self.show_segmented_image(self.imager.get_current_image())
        
    def compute_compliance_action(self):  # Function called when Calculate EF BT is pressed
        figure = plt.Figure(figsize=(6,6))
        if not self.split_quarters_status.get():
            plot = figure.add_subplot(111)
            plot.plot(self.imager.area, color="blue")
            plot.set_ylim(np.min(self.imager.area)-200, np.max(self.imager.area)+200)
            plot.set_xlabel('Slice')
            plot.set_ylabel('Area (mm^2)')
        else:
            offset = 20
            plot = figure.add_subplot(411)
            plot.plot(self.imager.areaquart[:,0], color="red", label="Posterior")
            plot.legend(loc="upper right")
            plot.set_ylim(np.min(self.imager.areaquart[:,0])-offset, np.max(self.imager.areaquart[:,0])+offset)
            plot.axes.get_xaxis().set_visible(False)

            plot = figure.add_subplot(412)
            plot.plot(self.imager.areaquart[:,1], color="green", label="Lateral")
            plot.legend(loc="upper right")
            plot.set_ylim(np.min(self.imager.areaquart[:,1])-offset, np.max(self.imager.areaquart[:,1])+offset)
            plot.axes.get_xaxis().set_visible(False)

            plot = figure.add_subplot(413)
            plot.plot(self.imager.areaquart[:,2], color="blue", label="Anterior")
            plot.legend(loc="upper right")
            plot.set_ylim(np.min(self.imager.areaquart[:,2])-offset, np.max(self.imager.areaquart[:,2])+offset)
            plot.axes.get_xaxis().set_visible(False)

            plot = figure.add_subplot(414)
            plot.plot(self.imager.areaquart[:,3], color="orange", label="Medial")
            plot.set_ylim(np.min(self.imager.areaquart[:,3])-offset, np.max(self.imager.areaquart[:,3])+offset)
            plot.legend(loc="upper right")

            plot.set_xlabel('Slice')
            plot = figure.add_subplot(111, frameon=False)
            plot.set_ylabel('Area (mm^2)', labelpad=20)
            plot.tick_params(axis='both', labelsize=0 ,length=0)
            
        self.figure = figure
        FigureCanvasTkAgg(figure, self.parent).get_tk_widget().grid(row=0, column=1, sticky=N)
        
        areas = self.imager.area
        min_area = np.min(areas[np.nonzero(areas)])
        max_area = np.max(areas[np.nonzero(areas)])
        self.compliance[0] = (max_area - min_area) / (int(self.systolic_pressure.get()) - int(self.diastolic_pressure.get()))
        
        for i in range(1,5): 
            quarter_areas = self.imager.areaquart[:, i-1]            
            if np.count_nonzero(quarter_areas) > 0:
                min_quarter_area = np.min(quarter_areas[np.nonzero(quarter_areas)])
                max_quarter_area = np.max(quarter_areas[np.nonzero(quarter_areas)])
                self.compliance[i] = (max_quarter_area - min_quarter_area) / (int(self.systolic_pressure.get()) - int(self.diastolic_pressure.get()))
        
        for i in range(0,4):
            quarter_perimeter = self.imager.local_perimeters[:, i]
            # if np.count_nonzero(quarter_perimeter) > 0:    
            #     min_perimeter = np.min(quarter_perimeter[np.nonzero(quarter_perimeter)])
            #     max_perimeter = np.max(quarter_perimeter[np.nonzero(quarter_perimeter)])
            self.strain[i] = (quarter_perimeter.max() - quarter_perimeter.min()) / quarter_perimeter.min()  
            
            # self.compliance[2] = (self.imager.areaquart[:,1].max() - self.imager.areaquart[:,1].min()) / (int(self.systolic_pressure.get()) - int(self.diastolic_pressure.get()))
            # self.compliance[3] = (self.imager.areaquart[:,2].max() - self.imager.areaquart[:,2].min()) / (int(self.systolic_pressure.get()) - int(self.diastolic_pressure.get()))
            # self.compliance[4] = (self.imager.areaquart[:,3].max() - self.imager.areaquart[:,3].min()) / (int(self.systolic_pressure.get()) - int(self.diastolic_pressure.get()))

        if not self.split_quarters_status.get():
            self.result.set(f" Global compliance: {self.compliance[0]:.4f}")
        else:
            # self.result.set(f" Global compliance: {self.compliance[0]:.4f} \n Avg, Min, Max Perimeters: {self.imager.perimeters.mean():.1f}, {self.imager.perimeters.min():.1f}, {self.imager.perimeters.max():.1f} \n Posterior: {self.compliance[1]:.4f} \n Lateral: {self.compliance[2]:.4f} \n Anterior: {self.compliance[3]:.4f} \n Medial: {self.compliance[4]:.4f}")
            self.result.set(f" Global compliance: {self.compliance[0]:.4f} \n \
                            Posterior compliance: {self.compliance[1]:.4f} \n \
                            Lateral compliance: {self.compliance[2]:.4f} \n \
                            Anterior compliance: {self.compliance[3]:.4f} \n \
                            Medial compliance: {self.compliance[4]:.4f} \n \n \
                            Posterior strain: {self.strain[0]} \n \
                            Lateral strain: {self.strain[1]} \n \
                            Anterior strain: {self.strain[2]} \n \
                            Medial strain: {self.strain[3]}")\

        self.result_label.grid(row=0, column=1, sticky='s')

    def show_contours_action(self): # Checker tickbox
        self.imager.contours(self.show_contours_status.get())
        self.show_image(self.imager.get_current_image())

    def split_quarters_action(self): # Checker tickbox
        self.imager.quarters(self.split_quarters_status.get())
        self.show_image(self.imager.get_current_image())              
        
    def segment_action(self):    # Function called when Apply SegNet BT is pressed
        self.progress_bar = Progressbar(self.parent, orient=HORIZONTAL, length=self.imager.get_slice_cnt(), mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky='sew', padx=(250, 50), ipady=5)
        self.progress_bar_status = StatusBar(self.parent)
        self.progress_bar_status.grid(row=1, column=0, sticky='se', padx=(50, 0))
        self.status.set("Segmenting Aorta...")
        
        predictions = np.zeros((self.volume.shape[0], self.volume.shape[1], self.volume.shape[2]), dtype=np.float32)
        # interpolated_predictions = np.zeros((self.volume.shape[0], self.volume.shape[1], self.volume.shape[2]), dtype=np.float32)
        for i in range(self.volume.shape[0]):
            self.parent.config(cursor="wait")
            self.parent.update()
            predictions[i,:,:] = self.model.get_segmentation(self.volume[i,:,:,:])
            
            self.progress_bar_status.set(str(i+1) + "/" + str(self.imager.get_slice_cnt()))
            self.progress_bar['value'] = (i+1) * (100 / self.imager.get_slice_cnt())
            
            # if (i == 0) or (i == self.volume.shape[0]-1):
            #     interpolated_predictions[i,:,:] = predictions[i,:,:]
            #     continue
            # else:
            #     interpolated_predictions[i,:,:] = interpolate(predictions[i-1,:,:], predictions[i+1,:,:], 1)
            #     print(f"i:{i}, mask shape: {interpolated_predictions[i,:,:].shape, interpolated_predictions[i,:,:].dtype}")
            #     # cv2.imshow("Mask interpolated", interpolated_predictions[i,:,:])
            #     # cv2.waitKey(0)
        
        if hasattr(self, 'progress_bar'): self.progress_bar.grid_remove()
        if hasattr(self, 'progress_bar_status'): self.progress_bar_status.grid_remove()
        
        self.vol_seg = self.imager.segmentation(predictions)
        self.imager.contours(self.show_contours_status.get())
        self.show_image(self.imager.get_current_image())
        self.compute_compliance_btn.grid(row=1, column=1, sticky="n")

        self.systolic_pressure_entry.grid(row=1, column=1, sticky=('sw'), ipadx=5, padx=150)
        self.systolic_pressure_label.grid(row=1, column=1, sticky=('sw'))
        self.diastolic_pressure_entry.grid(row=1,column=1,sticky=('se'), ipadx=5)
        self.diastolic_pressure_label.grid(row=1, column=1, sticky=('se'), padx=50)

        self.status.set("Segmentation Done!")
        FigureCanvasTkAgg(plt.Figure(figsize=(6,6)), self.parent).get_tk_widget().grid(row=0, column=1, sticky='n')
        self.manual_segmentation_btn.grid(row=1, column=2, sticky=(N))
        self.del_sel = Button(self.parent, text="Delete selected", command=self.delete_selected)
        self.listbox = Listbox(self.parent)
        
        self.show_contours_checkbtn.grid(row=1, column=1, sticky="sw", pady=50, padx=50)        
        self.convex_hull_checkbtn.grid(row=1, column=1, sticky="s", pady=50, padx=50)        
        # self.cog_checkbtn.grid(row=1, column=1, sticky="s", pady=0, padx=50)        
        self.split_quarters_checkbtn.grid(row=1, column=1, sticky="se", pady=50, padx=50)
        self.delete_slice_btn.grid(row=1, column=0, sticky='s', pady=25)
        
        self.parent.config(cursor="")
        
        
    def cog_action(self): # Checker tickbox
        copy_seg = copy.deepcopy(self.vol_seg)
        self.imager.segmentation(copy_seg, convex_hull=self.convex_hull_status.get(), cog=self.correct_cog_status.get())
        self.show_image(self.imager.get_current_image())
        
    def convex_hull_action(self):
        copy_seg = copy.deepcopy(self.vol_seg)
        self.imager.segmentation(copy_seg, convex_hull=self.convex_hull_status.get(), cog=self.correct_cog_status.get())
        self.show_image(self.imager.get_current_image())

    def delete_selected(self):
        self.listbox.delete('anchor')
        self.show_segmented_image(self.imager.get_current_image())
        
    def show_segmented_image(self, numpy_array):
        self.image = PIL.Image.fromarray(numpy_array)#.resize((np.asarray(numpy_array.shape[1])*2, np.asarray(numpy_array.shape[0])*2)) # This is changed because original ones were too small (we could resize them keeping the aspect ratio)
        width, height = self.image.size
        if self.image_id:
            self.canvas.delete(self.image_id)
            self.image_id = None
            self.canvas.imagetk = None
        new_size = int(self.imscale * width), int(self.imscale * height)
        self.photo = PIL.ImageTk.PhotoImage(self.image.resize(new_size)) # Another conversion needed
        self.image_id = self.canvas.create_image((0,0), anchor="nw", image=self.photo) # Show the image
        for i in range (self.listbox.size()):
            x = self.listbox.get(i)[0]*self.imscale
            y = self.listbox.get(i)[1]*self.imscale
            if self.listbox.curselection():
                if i == self.listbox.curselection()[0]:
                    self.canvas.create_rectangle(x-5, y-5 ,x , y, fill='yellow')
                else:
                    self.canvas.create_rectangle(x-5, y-5 ,x , y, fill='red')
            else:
                self.canvas.create_rectangle(x-5, y-5 ,x , y, fill='red')
                
    def manual_segmentation_action(self):
        self.canvas.unbind('<ButtonPress-1>')
        self.canvas.unbind('<B1-Motion>')
        self.canvas.bind('<Button-1>', self.get_coords)
        self.listbox.bind("<<ListboxSelect>>", self.listbox_sel)
        self.imager.clear_curr_seg()
        self.split_quarters_status.set(False)

        self.imager.quarters(0)
        self.convex_hull_status.set(False)
        self.show_segmented_image(self.imager.get_current_image())
        self.del_sel.grid(row=0, column=2, sticky=('n'), pady=50)
        self.listbox.grid(row=0, column=2, sticky=('n'), pady=100)
        self.manual_segmentation_end_btn.grid(row=0, column=2, sticky=('s'))
    
    def listbox_sel(self, event):
        self.show_segmented_image(self.imager.get_current_image())

    def manual_segmentation_end_action(self):
        self.del_sel.grid_remove()
        self.listbox.grid_remove()
        self.manual_segmentation_end_btn.grid_remove()
        coords = []

        for i in range (self.listbox.size()):
            coords.append(tuple((self.listbox.get(i))))
        
        if len(coords) >= 3:
            width, height = self.imager.get_current_image().shape[1], self.imager.get_current_image().shape[0]
            maskIm = PIL.Image.new('1', (width, height))
            PIL.ImageDraw.Draw(maskIm).polygon(coords, outline=1, fill=1)
            maskIm = maskIm.resize((width, height), resample=PIL.Image.ANTIALIAS)
            maskIm = np.array(maskIm)
            # print(type(maskIm), maskIm.shape, maskIm.max())
            
            # img = np.zeros((height, width, 3), dtype=np.uint8)
            # points = np.array(coords, dtype=np.int32)
            # points = points.reshape((-1, 1, 2))
            # import cv2
            # cv2.fillPoly(img, [points], (0,0,255), cv2.LINE_AA)
            # # cv2.polylines(img, [points], True, (0,0,255), 1, cv2.LINE_AA)
            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # (thresh, img_binary) = cv2.threshold(img_gray, 1, 255, cv2.THRESH_BINARY)
            # img_binary = np.where(img_binary>0, True, False)
            # # cv2.imshow("Image. Binary", img_binary)
            # # cv2.waitKey(0) 
            # print(img_binary.shape, img_binary.max())
            
            self.imager.update_seg(maskIm)
            self.vol_seg = copy.deepcopy(self.imager.get_segmentation())
            self.listbox.delete(0, 'end')
            self.canvas.delete("all")
            self.show_image(self.imager.get_current_image())
            self.compute_compliance_action()
        else:
            self.listbox.delete(0, 'end')
            self.canvas.delete("all")
            self.imager.reset_curr_seg()
            self.show_segmented_image(self.imager.get_current_image())
        
        self.listbox.unbind('<<ListboxSelect>>')
        self.canvas.unbind('<Button-1>')
        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)
    
    def get_coords(self, event):
        x_coord = self.canvas.canvasx(event.x)/self.imscale
        y_coord = self.canvas.canvasy(event.y)/self.imscale
        self.listbox.insert(END, [round(x_coord, 2), round(y_coord,2)])
        self.show_segmented_image(self.imager.get_current_image()) 

    def init_toolbar(self):     # Creating the MENU
        # Top level menu
        menubar = Menu(self.parent, bd=0)
        self.parent.config(menu=menubar, bd=0)
        # "File" menu
        filemenu = Menu(menubar, tearoff=False, bd=0)  # tearoff False removes the dashed line
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open DICOM Sequence", command=self.on_load_dicom)
        # Rest of the "File" menu
        filemenu.add_separator()
        filemenu.add_command(label="Save Segmentations", command=self.save_segmentation)
        filemenu.add_command(label="Exit", command=self.on_exit)
        
    def show_image(self, numpy_array): # Function to create Canvas for Systolic Volume
        if numpy_array is None:
            return
        # Convert numpy array into a PhotoImage (type needed for canvas) and add it to canvas
        self.image = PIL.Image.fromarray(numpy_array)#.resize((np.asarray(numpy_array.shape[1])*2, np.asarray(numpy_array.shape[0])*2)) # This is changed because original ones were too small (we could resize them keeping the aspect ratio)
        width, height = self.image.size
        if self.image_id is None:
            self.imscale = np.amin([self.parent.bbox(0,0)[2]/width, self.parent.bbox(0,0)[3]/height])
        else:
            self.canvas.delete(self.image_id)
            self.image_id = None
            self.canvas.imagetk = None

        new_size = int(self.imscale * width), int(self.imscale * height)
        self.photo = PIL.ImageTk.PhotoImage(self.image.resize(new_size)) # Another conversion needed
        self.image_id = self.canvas.create_image((0,0), anchor="nw", image=self.photo) # Show the image
        self.canvas.lower(self.image_id)
        self.canvas.imagetk = self.image  # keep an extra reference to prev_btnent garbage-collection

        self.slice_cnt.set(str(int(self.imager.index)+1)+"/"+str(self.imager.get_slice_cnt()))

    def previous_action(self): # If left key pressed
        move = -1
        self.keys(move)
    
    def next_action(self): # If right click pressed
        move = 1
        self.keys(move)
    
    def keys(self, move): # Same as the scroll 
        if (self.imager.index == 0) & (move<0):
            self.imager.index=self.imager.index
        elif (self.imager.index==(self.imager.get_slice_cnt()-1)) & (move>0):
            self.imager.index=self.imager.index
        else:
            self.imager.index += move
        self.slider.set(self.imager.index+1)
            
    def load_dicom(self, folder):
        self.clear_window()
        for root,_, files in (os.walk(folder)):
            shape = dicom.dcmread(os.path.join(root,files[0])).pixel_array.shape
            frames = len(os.listdir(folder))
            vol_or = np.zeros(shape=(frames,)+shape+(3,))
            files.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                        for x in re.findall(r'[^0-9]|[0-9]+', var)])
        for c,file1 in enumerate(files):
            if c == 0:
                ds = dicom.dcmread(os.path.join(root,file1))
            im = dicom.dcmread(os.path.join(root,file1)).pixel_array
            im = (im-im.min()) / (im.max() - im.min())
            vol_or[c,:,:,0]=vol_or[c,:,:,1]=vol_or[c,:,:,2] = im
        return vol_or, ds
    
    def clear_window(self):
        if self.imager is not None:
            FigureCanvasTkAgg(plt.Figure(figsize=(6,6)), self.parent).get_tk_widget().grid(row=0, column=1, sticky='n')
        
        self.show_contours_status.set(False)
        self.split_quarters_status.set(False)
        self.convex_hull_status.set(False)
        self.correct_cog_status.set(False)
        self.compliance = np.zeros(shape=(5,))
        self.systolic_pressure.set('130')
        self.diastolic_pressure.set('80')
        self.result.set('')
        
        if hasattr(self, 'import_btn'): self.import_btn.grid_remove()
        if hasattr(self, 'exit_btn'): self.exit_btn.grid_remove()
        if hasattr(self, 'cog_checkbtn'): self.cog_checkbtn.grid_remove()
        if hasattr(self, 'convex_hull_checkbtn'): self.convex_hull_checkbtn.grid_remove()
        if hasattr(self, 'compute_compliance_btn'): self.compute_compliance_btn.grid_remove()
        if hasattr(self, 'split_quarters_btn'): self.split_quarters_btn.grid_remove()
        if hasattr(self, 'show_contours_checkbtn'): self.show_contours_checkbtn.grid_remove() 
        if hasattr(self, 'split_quarters_checkbtn'): self.split_quarters_checkbtn.grid_remove()
        if hasattr(self, 'delete_slice_btn'): self.delete_slice_btn.grid_remove()
        if hasattr(self, 'result_label'): self.result_label.grid_remove()
        if hasattr(self, 'systolic_pressure_label'): self.systolic_pressure_label.grid_remove()
        if hasattr(self, 'systolic_pressure_entry'): self.systolic_pressure_entry.grid_remove()
        if hasattr(self, 'diastolic_pressure_label'): self.diastolic_pressure_label.grid_remove()
        if hasattr(self, 'diastolic_pressure_entry'): self.diastolic_pressure_entry.grid_remove()
        if hasattr(self, 'progress_bar'): self.progress_bar.grid_remove()
        if hasattr(self, 'progress_bar_status'): self.progress_bar_status.grid_remove()
        
    def on_load_dicom(self): # When Open Systolic Volume in the menu is selected
        filename = askdirectory()
        self.filename = os.path.basename(os.path.normpath(filename))
        datasets = self.load_dicom(filename) # Read the nifti file in the same
        # way as in the colab code. IMPORTANT only 3d nifti. We should add a checker and update the
        # status bar    
        self.volume = np.array(datasets[0])     # Store the volume
        self.imager = Imager(datasets) # Pass the volume to the imager function
        self.status.set("Opened DICOM files")
        self.slice_cnt.grid(row=1, column=0, sticky='n', pady=50)
        self.canvas = Canvas(self.parent, highlightthickness=0, width=self.parent.winfo_screenwidth()/2)
        self.canvas.grid(row=0, column=0, sticky='nswe')

        self.canvas.bind('<ButtonPress-1>', self.move_from)
        self.canvas.bind('<B1-Motion>',     self.move_to)

        self.imscale = 1.0
        self.image_id = None
        self.delta = 1.3     
        
        self.segment_btn.grid(row=1, column=0, sticky=None)
        self.prev_btn.grid(row=1, column=0, sticky=NW, pady=50)
        self.next_btn.grid(row=1, column=0, sticky=NE, pady=50)
        self.show_image(self.imager.get_current_image())
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        self.slider = Scale(self.parent, from_=1, to=self.imager.get_slice_cnt(), command=self.update_index, orient='horizontal')
        self.slider.grid(row=1, column=0, sticky='ewn')
        
    
    def update_index(self, event):
        self.imager.index = self.slider.get()-1 # In any other case, hange the index accordingly
        self.show_image(self.imager.get_current_image())

    def move_from(self, event):
        ''' Remember prev_btnious coordinates for scrolling with the mouse '''
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def save_segmentation(self): # When Save Segmentations in the menu is selected
        if np.sum(self.imager._segmentation)==0: # If no segmentation exists in Volume 1
            self.status.set("Retrieve the segmentations first.")
        else:
            filename = askdirectory()
            self.status.set("Saving segmentations...")
            image = nib.Nifti1Image(self.imager.get_segmentation().T, np.eye(4))
            name = self.filename + '_seg.nii.gz'
            #image.to_filename(os.path.join(filename, name))
            nib.save(image,os.path.join(filename, name))
            self.status.set("Segmentations saved")

            df = pd.DataFrame(np.concatenate((self.imager.areaquart, np.reshape(self.imager.area, self.imager.area.shape+(1,))),axis=1), columns=['Posterior', 'Lateral', 'Anterior', 'Medial', 'Global'])
            df.to_csv(os.path.join(filename, self.filename + '_compliance_curves.csv'), index=False, header=True)
            
            data = np.array([ 
                [self.compliance[0]], [self.compliance[1]], [self.compliance[2]], [self.compliance[3]], [self.compliance[4]], 
                [self.imager.area.min()], [self.imager.area.max()],  
                [self.strain[0]], [self.strain[1]], [self.strain[2]], [self.strain[3]], 
                             
            ]).T
            df = pd.DataFrame(data, columns=['global_compliance', 'posterior_compliance', 'lateral_compliance', 
                                             'anterior_compliance', 'medial_compliance', 'min_area_pred', 'max_area_pred', 
                                             'posterior_strain', 'lateral_strain', 
                                             'anterior_strain', 'medial_strain'])
            
            df.to_csv(os.path.join(filename, self.filename + '_results.csv'), index=False, header=True)
            
            self.figure.savefig(os.path.join(filename, self.filename + '_figure.png'))
            
    def on_exit(self):  # Close the UI
        self.parent.destroy()
        self.quit()
        
# Main definition of the program
def main():
    root = Tk() # Create GUI object
    root.state('zoomed')
    # w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    # root.minsize(600, 400)  # Define min size of the window
    # root.geometry("%dx%d+0+0" % (w, h))
    # root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))    # Set size to the full screen size
    MainWindow(root)  # Assign all the functions in mainwindow to the app
    root.mainloop() # Start the main loop that initializes the app

if __name__ == '__main__':
    main()