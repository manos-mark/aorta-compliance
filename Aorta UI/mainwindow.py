from tkinter.filedialog import askopenfilenames, askdirectory
from tkinter import messagebox as mbox
from tkinter import ttk
from nibabel.spatialimages import Header
import pydicom as dicom
import PIL.Image, PIL.ImageTk, os, sys, PIL.ImageDraw
from statusbar import *
from imager import Imager
import numpy as np
import pandas as pd
from sn_button import SegNet
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt 
import nibabel as nib
from sys import platform
import tkinter as tk
import re
import copy

class AutoScrollbar(ttk.Scrollbar):
    ''' A scrollbar that hides itself if it's not needed.
        Works only if you use the grid geometry manager '''
    def set(self, lo, hi):
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            self.grid_remove()
        else:
            self.grid()
        ttk.Scrollbar.set(self, lo, hi)

    def pack(self, **kw):
        raise tk.TclError('Cannot use pack with this widget')

    def place(self, **kw):
        raise tk.TclError('Cannot use place with this widget')

# Class to define structure, connections and widgets inside the GUI
class MainWindow(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent # This is the GUI object (root)
        # Systolic Volume will be subscripted with index 1
        self.image1 = None   # Define all the objects the GUI will contain
        self.photo1 = None   # To get the image that will be displayed using canvas
        self.canvas1 = None  # Object to display images
        self.imager1 = None  # Function to load the nifti file        
        # Diastolic Volume will be subscripted with index 2
        self.status = None  # Object to show some text at the bottom
        self.countim = None
        self.prev = None
        self.next = None
        self.right_click_menu = None # Object to enable actions using the right click
        self.volume1 = []   # To store both volumes
        self.name1 = ''     # To store file names for optional saving
        self.init_ui()

    def init_ui(self):
        self.parent.title("Aortic Segmentation and Compliance Tool")   # Title of the GUI
        
        # For development, considers the relative path in string
        # When build into .exe using pyinstaller, changes the directory
        # To the TEMP folder wherever the executable unpacks the necessary files
        # NOTE: this is only needed when using files apart from .py programs
        # In this program, the icon and TF model
        def resource_path(relative):
            try:
                base_path = sys._MEIPASS
            except Exception:
                base_path = os.path.abspath(".")
            return os.path.join(base_path, relative)
        
        #self.parent.iconbitmap(resource_path('bhicon.ico'))
        self.init_toolbar() # initialize the top menu
        # For .grid Widget management
        self.parent.columnconfigure([0,1], weight=3)
        self.parent.rowconfigure([0], weight=3)
        self.parent.rowconfigure([1], weight=1)
        # Image canvas 
        self.check = IntVar()
        self.quart = IntVar()
        self.ch = IntVar()
        self.cog = IntVar()
        self.compliance = np.zeros(shape=(5,))
        self.sp = StringVar(value='130')
        self.dp = StringVar(value='80')
        self.result = StringVar()
        # Load the model and initiate it (avoid of doing it when aplying the model)
        self.model = SegNet()
        # Message Function
        mbox.showinfo("Usage Information", ("This program is used for segmentation of the ascending aorta and calculation "
"of the Aortic Compliance. The segmentation is based on UNet Convolutional Neural Network." 
"\n\nThe user should provide the application with a cardiac MRI sequence in DICOM format  by going to "
"File -> Open DICOM Sequence.\n\nThe program will showcase the images provided. Better visualization can be achieved by zooming in/out and panning."
" By clicking on the button \"Segment Aorta\", the "
"segmentation will be shown. The user can manually segment one frame, once the aorta is correctly zoomed, by clicking on the \"Segment manually\" "
"button. The segmentation contour should be defined by left-clicking on the aorta contour. Specific points can be removed by selecting them on the list "
"and clicking on \"Delete selected\". Once all the contour points are defined, the user should click on \"End segmentation\". The segmentation will be stored"
" and the compliance recomputed."
"\n\nSystolic and Diastolic pressures can be entered by the user (by default are fixed to 130 and 80 mmHg, "
"respectively).\n\nBy clicking on the button \"Compute Compliance\", the global compliance together with the local compliances "
"(if \"Quarters\" is selected, will be shown"
" together with area of the segmented aorta over time.\nSegmentations together with the compliance results can "
"be saved in File -> Save Segmentation."))


        # Status bar
        self.status = StatusBar(self.parent)
        self.status.grid(row = 1, column = 0, sticky = (S, W))
        self.countim = StatusBar(self.parent)
        self.prev = Button(self.parent,text = '< Previous', command = self.left_action)
        self.next = Button(self.parent,text = 'Next >', command = self.right_action)
        # Right-click menu
        self.right_click_menu = Menu(self.parent, tearoff=0) # Define the menu when the rc is pressed
        self.right_click_menu.add_command(label="Exit", command=self.on_exit)        
        # SegNet button
        self.sn_bt = Button(self.parent,text = 'Segment Aorta', command = self.bt_action)        
        # EF button
        self.ef_bt = Button(self.parent,text = 'Compute Compliance', command = self.calc_ef)

        self.manual_seg = Button(self.parent,text = 'Segment Manually', command = self.man_seg)
        self.manual_seg_end = Button(self.parent,text = 'End Segmentation', command = self.man_seg_end)
        #self.canvas1.focus_set() # Function used to listen to the keyboard continuously to control the images
        #self.canvas1.bind("<Left>", self.left_action) # Bind left key interaction to canvas and when pressed do self.left_action function
        #self.canvas1.bind("<Right>", self.right_action)
        #self.canvas1.bind("<Button-3>", self.show_right_click_menu)  # <Button-3> is the right click event
        #self.canvas1.bind("<MouseWheel>", self.scroll_images)
        # Titles of Canvas
        self.labelSP = Label(self.parent, text="Systolic Pressure: ", relief=(RAISED))
        self.labelDP  = Label(self.parent, text="Diastolic Pressure: ", relief=(RAISED))
        self.labelresult = Label(self.parent, textvariable=self.result, relief=(RAISED))
        self.entrySP = Entry(self.parent, textvariable=self.sp, width=2)
        self.entryDP = Entry(self.parent, textvariable=self.dp, width=2)
        # Additional variables to control the Workflow
        # To use either 1 or 2 Volumes
        self.vol1load = 0   # 1 when volume is loaded
        self.vol1seg = 0    # 1 when volume is segmented 


        # Area canvas
        #figure = Figure(figsize=(5, 4), dpi=100)
        #plot = figure.add_subplot(1, 1, 1)
        #self.canvas2 = FigureCanvasTkAgg(figure, self.parent)
        #self.canvas2.get_tk_widget().grid(row=1, column=1)
        # Variable that is either 0 or 1 whenever the checker is preesed     
        
    def calc_ef(self):  # Function called when Calculate EF BT is pressed
        figure = plt.Figure(figsize=(5,5))
        if (self.quart.get() == 0):
            plot = figure.add_subplot(111)
            plot.plot(self.imager1.area, color="blue")
            plot.set_xlabel('Frame')
            plot.set_xlabel('Area (mm^2)')
        else:
            plot = figure.add_subplot(411)
            plot.plot(self.imager1.areaquart[:,0], color="red", label="Posterior")
            plot.legend(loc="upper right")
            plot.axes.get_xaxis().set_visible(False)
            #plot.axes.get_yaxis().set_visible(False)
            plot = figure.add_subplot(412)
            plot.plot(self.imager1.areaquart[:,1], color="green", label="Lateral")
            plot.legend(loc="upper right")
            plot.axes.get_xaxis().set_visible(False)
            #plot.axes.get_yaxis().set_visible(False)
            plot = figure.add_subplot(413)
            plot.plot(self.imager1.areaquart[:,2], color="blue", label="Anterior")
            plot.legend(loc="upper right")
            plot.axes.get_xaxis().set_visible(False)
            #plot.axes.get_yaxis().set_visible(False)
            plot = figure.add_subplot(414)
            plot.plot(self.imager1.areaquart[:,3], color="yellow", label="Medial")
            plot.legend(loc="upper right")
            #plot.axes.get_xaxis().set_visible(False)
            #plot.axes.get_yaxis().set_visible(False)
            plot.set_xlabel('Frame')
            plot = figure.add_subplot(111, frameon=False)
            plot.set_ylabel('Area (mm^2)', labelpad=20)
            plot.tick_params(axis='both', labelsize=0 ,length=0)
            
        
        FigureCanvasTkAgg(figure, self.parent).get_tk_widget().grid(row=0, column=1)
        self.compliance[0] = (self.imager1.area.max() - self.imager1.area.min()) / (int(self.sp.get()) - int(self.dp.get()))
        self.compliance[1] = (self.imager1.areaquart[:,0].max() - self.imager1.areaquart[:,0].min()) / (int(self.sp.get()) - int(self.dp.get()))
        self.compliance[2] = (self.imager1.areaquart[:,1].max() - self.imager1.areaquart[:,1].min()) / (int(self.sp.get()) - int(self.dp.get()))
        self.compliance[3] = (self.imager1.areaquart[:,2].max() - self.imager1.areaquart[:,2].min()) / (int(self.sp.get()) - int(self.dp.get()))
        self.compliance[4] = (self.imager1.areaquart[:,3].max() - self.imager1.areaquart[:,3].min()) / (int(self.sp.get()) - int(self.dp.get()))
        if (self.quart.get() == 0):
            self.result.set(f"Global compliance: {self.compliance[0]:.4f}")
        else:
            self.result.set(f"Global compliance: {self.compliance[0]:.4f}\nPosterior: {self.compliance[1]:.4f}\nLateral: {self.compliance[2]:.4f}\nAnterior: {self.compliance[3]:.4f}\nMedial: {self.compliance[4]:.4f}")
        self.labelresult.grid(column=1, row=1)

    def check_action(self): # Checker tickbox
        self.imager1.contours(self.check.get())
        self.show_image1(self.imager1.get_current_image())

    def quart_action(self): # Checker tickbox
        self.imager1.quarters(self.quart.get())
        self.show_image1(self.imager1.get_current_image())              
        
    def bt_action(self):    # Function called when Apply SegNet BT is pressed
        self.status.set("Segmenting Aorta...")
        self.vol_seg = self.imager1.segmentation(self.model.get_segmentation(self.volume1))
        self.imager1.contours(self.check.get())
        self.show_image1(self.imager1.get_current_image())
        self.vol1seg = 1
        self.ef_bt.grid(column=1, row=1, sticky= (W))
        # Place the checker to Show Contour on UI
        Checkbutton(self.parent, text="Show contour", variable=self.check, command = self.check_action).grid(row=1, column=0,sticky="sw", pady=50, padx=50)
        Checkbutton(self.parent, text="Quarters", variable=self.quart, command = self.quart_action).grid(row=1, column=0, sticky="se", pady=50, padx=50)
        self.labelSP.grid(row=1, column = 2, sticky=(E))
        self.labelDP.grid(row=1, column=2, sticky=(SE), pady=25)
        self.entryDP.grid(row=1,column=3,sticky=(SW), ipadx=5, pady=25)
        self.entrySP.grid(row=1, column=3, sticky=(W), ipadx=5)
        self.status.set("Segmentation Done!")
        self.manual_seg.grid(row=0, column=2, sticky = (N))
        self.del_sel = tk.Button(self.parent, text="Delete selected", command=self.del_selected)
        self.listbox = tk.Listbox(self.parent)
        self.ch_ch =  Checkbutton(self.parent, text="Apply Convex Hull", variable=self.ch, command=self.ch_action)
        self.ch_ch.grid(row=1, column=0, sticky="s", pady=50, padx=50)
        self.cog_bt =  Checkbutton(self.parent, text="Correct c.o.g.", variable=self.cog, command=self.cog_action)
        self.cog_bt.grid(row=1, column=0, sticky="s", pady=25, padx=50)

    def cog_action(self): # Checker tickbox
        copy_seg = copy.deepcopy(self.vol_seg)
        self.imager1.segmentation(copy_seg, ch=self.ch.get(), cog=self.cog.get())
        self.show_image1(self.imager1.get_current_image())
        
    def ch_action(self):
        copy_seg = copy.deepcopy(self.vol_seg)
        self.imager1.segmentation(copy_seg, ch=self.ch.get())
        self.show_image1(self.imager1.get_current_image())

    def del_selected(self):
        self.listbox.delete(tk.ANCHOR)
        self.show_image_seg(self.imager1.get_current_image())
        
    def show_image_seg(self, numpy_array):
        self.image1 = PIL.Image.fromarray(numpy_array)#.resize((np.asarray(numpy_array.shape[1])*2, np.asarray(numpy_array.shape[0])*2)) # This is changed because original ones were too small (we could resize them keeping the aspect ratio)
        width, height = self.image1.size
        if self.imageid:
            self.canvas1.delete(self.imageid)
            self.imageid = None
            self.canvas1.imagetk = None
        new_size = int(self.imscale * width), int(self.imscale * height)
        self.photo1 = PIL.ImageTk.PhotoImage(self.image1.resize(new_size)) # Another conversion needed
        self.imageid = self.canvas1.create_image((0,0), anchor="nw", image=self.photo1) # Show the image
        for ind in range (self.listbox.size()):
            x = self.listbox.get(ind)[0]*self.imscale
            y = self.listbox.get(ind)[1]*self.imscale
            if self.listbox.curselection():
                if ind == self.listbox.curselection()[0]:
                    self.canvas1.create_rectangle(x-5, y-5 ,x , y, fill='yellow')
                else:
                    self.canvas1.create_rectangle(x-5, y-5 ,x , y, fill='red')
            else:
                self.canvas1.create_rectangle(x-5, y-5 ,x , y, fill='red')
                
    def man_seg(self):
        self.canvas1.unbind('<ButtonPress-1>')
        self.canvas1.unbind('<B1-Motion>')
        self.canvas1.unbind('<MouseWheel>')  # with Windows and MacOS, but not Linux
        self.canvas1.unbind('<Button-5>')  # only with Linux, wheel scroll down
        self.canvas1.unbind('<Button-4>')  # only with Linux, wheel scroll up
        self.canvas1.bind('<Button-1>', self.get_coords)
        self.listbox.bind("<<ListboxSelect>>", self.listbox_sel)
        self.imager1.del_curr_seg()
        self.quart.set(0)

        self.imager1.quarters(0)
        self.ch.set(0)
        self.show_image_seg(self.imager1.get_current_image())
        self.del_sel.grid(row=0, column=2, sticky=(N), pady=50)
        self.listbox.grid(row=0, column=2, sticky=(N), pady=100)
        self.manual_seg_end.grid(row=0, column=2, sticky=(S))
        #print(self.canvas1.index(self.imageid))
    
    def listbox_sel(self, event):
        self.show_image_seg(self.imager1.get_current_image())

    def man_seg_end(self):
        self.del_sel.grid_remove()
        self.listbox.grid_remove()
        self.manual_seg_end.grid_remove()
        coords = []
        seg = np.zeros(shape=self.imager1.get_current_image().shape)
        for ind in range (self.listbox.size()):
            coords.append(tuple((self.listbox.get(ind))))
        maskIm = PIL.Image.new('1', (self.imager1.get_current_image().shape[1], self.imager1.get_current_image().shape[0]))
        PIL.ImageDraw.Draw(maskIm).polygon(coords, outline=1, fill=1)
        maskIm = np.array(maskIm)
        self.imager1.update_seg(maskIm)
        self.vol_seg = copy.deepcopy(self.imager1.get_segmentation())
        self.listbox.delete(0, 'end')
        self.canvas1.delete("all")
        self.show_image1(self.imager1.get_current_image())
        self.calc_ef()
        self.listbox.unbind('<<ListboxSelect>>')
        self.canvas1.unbind('<Button-1>')
        self.canvas1.bind('<ButtonPress-1>', self.move_from)
        self.canvas1.bind('<B1-Motion>',     self.move_to)
        self.canvas1.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas1.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.canvas1.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
    
    def get_coords(self, event):
        #print(event.x, event.y)
        x = self.canvas1.canvasx(event.x)
        y = self.canvas1.canvasy(event.y)
        x_coord = self.canvas1.canvasx(event.x)/self.imscale
        y_coord = self.canvas1.canvasy(event.y)/self.imscale
        self.listbox.insert(tk.END, [round(x_coord, 2), round(y_coord,2)])
        self.show_image_seg(self.imager1.get_current_image()) 

    def init_toolbar(self):     # Creating the MENU
        # Top level menu
        menubar = Menu(self.parent, bd=0)
        self.parent.config(menu=menubar, bd=0)
        # "File" menu
        filemenu = Menu(menubar, tearoff=False, bd=0)  # tearoff False removes the dashed line
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Open DICOM Sequence", command=self.on_open1)
        # Rest of the "File" menu
        filemenu.add_separator()
        filemenu.add_command(label="Save Segmentations", command=self.save_seg)
        filemenu.add_command(label="Exit", command=self.on_exit)

    def show_image1(self, numpy_array): # Function to create Canvas for Systolic Volume
        if numpy_array is None:
            return
        # Convert numpy array into a PhotoImage (type needed for canvas) and add it to canvas
        self.image1 = PIL.Image.fromarray(numpy_array)#.resize((np.asarray(numpy_array.shape[1])*2, np.asarray(numpy_array.shape[0])*2)) # This is changed because original ones were too small (we could resize them keeping the aspect ratio)
        width, height = self.image1.size
        if self.imageid is None:
            self.imscale = np.amin([self.parent.bbox(0,0)[2]/width, self.parent.bbox(0,0)[3]/height])
            new_size = int(self.imscale * width), int(self.imscale * height)
            self.photo1 = PIL.ImageTk.PhotoImage(self.image1.resize(new_size)) # Another conversion needed
            self.imageid = self.canvas1.create_image((0,0), anchor="nw", image=self.photo1) # Show the image
            self.canvas1.lower(self.imageid)
            self.canvas1.imagetk = self.image1  # keep an extra reference to prevent garbage-collection
        else:
            self.canvas1.delete(self.imageid)
            self.imageid = None
            self.canvas1.imagetk = None
            new_size = int(self.imscale * width), int(self.imscale * height)
            self.photo1 = PIL.ImageTk.PhotoImage(self.image1.resize(new_size)) # Another conversion needed
            self.imageid = self.canvas1.create_image((0,0), anchor="nw", image=self.photo1) # Show the image
            self.canvas1.lower(self.imageid)
            self.canvas1.imagetk = self.image1  # keep an extra reference to prevent garbage-collection

        self.countim.set(str(self.imager1.index+1)+"/"+str(self.imager1.get_num_im()))
        #print(self.canvas1.bbox(self.imageid))
        #print(new_size)

    def show_right_click_menu(self, e): # Function when RC is pressed
        self.right_click_menu.post(e.x_root, e.y_root) # Display the menu at the location of the mouse

    def scroll_images(self, e): # Scroll through images using the mouse wheel
    # e.delta is the event that happens when scrolling
    # if it's upwards e.delta = 120, downwards e.delta = -120
        if (sys.platform=='darwin'):
            e.delta = e.delta
        else:
            e.delta = e.delta/120
            
        if (self.imager1.index == 0) & (e.delta<0):
            # If we are on the 1st image and scroll downwards, don't change the index
            self.imager1.index=self.imager1.index
        elif (self.imager1.index==(self.imager1.get_num_im()-1)) & (e.delta>0):
            # If we are on the last image, and scroll upwards, don't continue
            self.imager1.index=self.imager1.index
        else:
            self.imager1.index += int(e.delta) # In any other case, hange the index accordingly
        self.show_image1(self.imager1.get_current_image()) # Show slice index of the volume
        
    def left_action(self): # If left key pressed
        move = -1
        self.keys(move)
    
    def right_action(self): # If right click pressed
        move = 1
        self.keys(move)
    
    def keys(self, move): # Same as the scroll 
        if (self.imager1.index == 0) & (move<0):
            self.imager1.index=self.imager1.index
        elif (self.imager1.index==(self.imager1.get_num_im()-1)) & (move>0):
            self.imager1.index=self.imager1.index
        else:
            self.imager1.index += move
        self.slider.set(self.imager1.index+1)
            
    def load_dicom(self, folder):
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

    def on_open1(self): # When Open Systolic Volume in the menu is selected
        filename = askdirectory()
        self.name1 = os.path.basename(os.path.normpath(filename))
        datasets = self.load_dicom(filename) # Read the nifti file in the same
        # way as in the colab code. IMPORTANT only 3d nifti. We should add a checker and update the
        # status bar    
        self.volume1 = datasets[0]     # Store the volume
        self.imager1 = Imager(datasets) # Pass the volume to the imager function
        self.status.set("Opened DICOM files")
        self.countim.grid(row = 1, column = 0, sticky = (N), pady=50)
        self.vol1load = 1
        self.vol1seg = 0
        self.vbar = AutoScrollbar(self.parent, orient='vertical')
        self.hbar = AutoScrollbar(self.parent, orient='horizontal')
        self.vbar.grid(row=0, column=0, sticky='nes')
        self.hbar.grid(row=0, column=0, sticky='ews')
        self.canvas1 = tk.Canvas(self.parent, highlightthickness=0, xscrollcommand=self.hbar.set, yscrollcommand=self.vbar.set, width=600)
        self.canvas1.grid(row=0, column=0, sticky='nswe')
        self.vbar.configure(command=self.canvas1.yview)  # bind scrollbars to the canvas
        self.hbar.configure(command=self.canvas1.xview)

        self.canvas1.bind('<ButtonPress-1>', self.move_from)
        self.canvas1.bind('<B1-Motion>',     self.move_to)
        self.canvas1.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas1.bind('<Button-5>',   self.wheel)  # only with Linux, wheel scroll down
        self.canvas1.bind('<Button-4>',   self.wheel)  # only with Linux, wheel scroll up
        #self.canvas1.bind('<Button-1>', self.getcoords)
        #self.text = self.canvas1.create_text(0, 0, anchor='nw', text='Scroll to zoom')

        self.imscale = 1.0
        self.imageid = None
        self.delta = 0.75        
        
        self.sn_bt.grid(row = 1, column = 0, sticky = None)
        self.prev.grid(row=1, column = 0, sticky = (NW), pady=50)
        self.next.grid(row=1, column=0, sticky=(NE), pady=50)
        #self.text = self.canvas1.create_text(0, 0, anchor='nw', text=' ')
        self.show_image1(self.imager1.get_current_image())
        self.canvas1.configure(scrollregion=self.canvas1.bbox('all'))
        self.slider = Scale(self.parent, from_=1, to=self.imager1.get_num_im(), command=self.update_ind, orient=HORIZONTAL)
        self.slider.grid(row=1, column=0, sticky='ewn')
    
    def update_ind(self, event):
        self.imager1.index = self.slider.get()-1 # In any other case, hange the index accordingly
        self.show_image1(self.imager1.get_current_image())

    def move_from(self, event):
        ''' Remember previous coordinates for scrolling with the mouse '''
        self.canvas1.scan_mark(event.x, event.y)

    def move_to(self, event):
        ''' Drag (move) canvas to the new position '''
        self.canvas1.scan_dragto(event.x, event.y, gain=1)

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
        x = self.canvas1.canvasx(event.x)
        y = self.canvas1.canvasy(event.y)
        self.canvas1.scale('all', x, y, scale, scale)
        self.show_image1(self.imager1.get_current_image())
        self.canvas1.configure(scrollregion=self.canvas1.bbox('all'))

    def save_seg(self): # When Save Segmentations in the menu is selected
        if np.sum(self.imager1._segmentation)==0: # If no segmentation exists in Volume 1
            self.status.set("Retrieve the segmentations first.")
        else:
            filename = askdirectory()
            self.status.set("Saving segmentations...")
            image = nib.Nifti1Image(self.imager1.get_segmentation().T, np.eye(4))
            name = self.name1 + '_seg.nii.gz'
            #image.to_filename(os.path.join(filename, name))
            nib.save(image,os.path.join(filename, name))
            self.status.set("Segmentations saved")
            f = open(os.path.join(filename, self.name1 + '_results.txt'), "w")
            f.write(self.result.get())
            f.close()
            df = pd.DataFrame(np.concatenate((self.imager1.areaquart, np.reshape(self.imager1.area, self.imager1.area.shape+(1,))),axis=1), columns=['Posterior', 'Lateral', 'Anterior', 'Medial', 'Global'])
            df.to_csv(os.path.join(filename, self.name1 + '_compliance_curves.csv'), index=False, header=True)
            
    def on_exit(self):  # Close the UI
        self.quit()