import copy
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt 
import PIL.Image, PIL.ImageTk, os, sys, PIL.ImageDraw
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from .header_widget import HeaderWidget
from .toolbar_widget import ToolbarWidget
from .statusbar_widget import StatusBarWidget

# Class to define structure, connections and widgets inside the GUI
class MainWindow(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)
        self.parent = parent # This is the GUI object (window)
        
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
          
        # Image canvas 
        self.check = tk.IntVar()
        self.quart = tk.IntVar()
        self.ch = tk.IntVar()
        self.cog = tk.IntVar()
        self.compliance = np.zeros(shape=(5,))
        self.sp = tk.StringVar(value='130')
        self.dp = tk.StringVar(value='80')
        self.result = tk.StringVar()

        # Load the model and initiate it (avoid of doing it when aplying the model)
        # self.model = SegNet()
        
        # Status bar
        self.status = StatusBarWidget(self.parent)
        self.status.grid(row = 1, column = 0, sticky = ('s', 'w'))
        self.countim = StatusBarWidget(self.parent)
        self.prev = tk.Button(self.parent,text = '< Previous', command = self.left_action)
        self.next = tk.Button(self.parent,text = 'Next >', command = self.right_action)

        # Right-click menu
        self.right_click_menu = tk.Menu(self.parent, tearoff=0) # Define the menu when the rc is pressed
        self.right_click_menu.add_command(label="Exit", command=self.on_exit)        

        # SegNet button
        self.sn_bt = tk.Button(self.parent,text = 'Segment Aorta', command = self.bt_action)        

        # EF button
        self.ef_bt = tk.Button(self.parent,text = 'Compute Compliance', command = self.calc_ef)

        self.manual_seg = tk.Button(self.parent,text = 'Segment Manually', command = self.man_seg)
        self.manual_seg_end = tk.Button(self.parent,text = 'End Segmentation', command = self.man_seg_end)

        # Titles of Canvas
        self.labelSP = tk.Label(self.parent, text="Systolic Pressure: ", relief='raised')
        self.labelDP  = tk.Label(self.parent, text="Diastolic Pressure: ", relief='raised')
        self.labelresult = tk.Label(self.parent, textvariable=self.result, relief='raised')
        self.entrySP = tk.Entry(self.parent, textvariable=self.sp, width=2)
        self.entryDP = tk.Entry(self.parent, textvariable=self.dp, width=2)

        # Additional variables to control the Workflow
        # To use either 1 or 2 Volumes
        self.vol1load = 0   # 1 when volume is loaded
        self.vol1seg = 0    # 1 when volume is segmented 

        toolbar = ToolbarWidget(self.parent, self)
        
        # three frames on top of each other
        header_frame = tk.Frame(self.parent, borderwidth=2, pady=2)
        header = HeaderWidget(header_frame, self)
        
        # center_frame = tk.Frame(self.parent, borderwidth=2, pady=5)
        # bottom_frame = tk.Frame(self.parent, borderwidth=2, pady=5)
        
        header_frame.grid(row=0, column=0)
        # center_frame.grid(row=1, column=0)
        # bottom_frame.grid(row=2, column=0)
                
        # For .grid Widget management
        # self.parent.columnconfigure([0,1], weight=3)
        # self.parent.rowconfigure([0], weight=3)
        # self.parent.rowconfigure([1], weight=1)
        
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
        
    def get_coords(self, event):
        #print(event.x, event.y)
        x = self.canvas1.canvasx(event.x)
        y = self.canvas1.canvasy(event.y)
        x_coord = self.canvas1.canvasx(event.x)/self.imscale
        y_coord = self.canvas1.canvasy(event.y)/self.imscale
        self.listbox.insert(tk.END, [round(x_coord, 2), round(y_coord,2)])
        self.show_image_seg(self.imager1.get_current_image()) 
        
    def listbox_sel(self, event):
        self.show_image_seg(self.imager1.get_current_image())
        
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
        self.del_sel.grid(row=0, column=2, sticky=('n'), pady=50)
        self.listbox.grid(row=0, column=2, sticky=('n'), pady=100)
        self.manual_seg_end.grid(row=0, column=2, sticky=('s'))
        #print(self.canvas1.index(self.imageid))
        
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
        
    def bt_action(self):    # Function called when Apply SegNet BT is pressed
        self.status.set("Segmenting Aorta...")
        self.vol_seg = self.imager1.segmentation(self.model.get_segmentation(self.volume1))
        self.imager1.contours(self.check.get())
        self.show_image1(self.imager1.get_current_image())
        self.vol1seg = 1
        self.ef_bt.grid(column=1, row=1, sticky= ('w'))
        # Place the checker to Show Contour on UI
        tk.Checkbutton(self.parent, text="Show contour", variable=self.check, command = self.check_action).grid(row=1, column=0,sticky="sw", pady=50, padx=50)
        tk.Checkbutton(self.parent, text="Quarters", variable=self.quart, command = self.quart_action).grid(row=1, column=0, sticky="se", pady=50, padx=50)
        self.labelSP.grid(row=1, column = 2, sticky=('e'))
        self.labelDP.grid(row=1, column=2, sticky=('se'), pady=25)
        self.entryDP.grid(row=1,column=3,sticky=('sw'), ipadx=5, pady=25)
        self.entrySP.grid(row=1, column=3, sticky=('w'), ipadx=5)
        self.status.set("Segmentation Done!")
        self.manual_seg.grid(row=0, column=2, sticky = ('n'))
        self.del_sel = tk.Button(self.parent, text="Delete selected", command=self.del_selected)
        self.listbox = tk.Listbox(self.parent)
        self.ch_ch =  tk.Checkbutton(self.parent, text="Apply Convex Hull", variable=self.ch, command=self.ch_action)
        self.ch_ch.grid(row=1, column=0, sticky="s", pady=50, padx=50)
        self.cog_bt =  tk.Checkbutton(self.parent, text="Correct c.o.g.", variable=self.cog, command=self.cog_action)
        self.cog_bt.grid(row=1, column=0, sticky="s", pady=25, padx=50)
        
    def on_exit(self):  # Close the UI
        self.parent.destroy()
        self.quit()
