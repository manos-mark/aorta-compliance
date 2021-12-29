import tkinter as tk
from tkinter.filedialog import askdirectory

from .base_widget import BaseIOWidget

class ToolbarWidget(BaseIOWidget):
    def __init__(self, root, main_window):
        tk.Frame.__init__(self, main_window)
        self.main_window = main_window
        self.root = root
        
        # Top level menu
        menubar = tk.Menu(self.root, bd=0)
        self.root.config(menu=menubar, bd=0)
        # "File" menu
        filemenu = tk.Menu(menubar, tearoff=False, bd=0)  # tearoff False removes the dashed line
        menubar.add_cascade(label="File", menu=filemenu)
        filemenu.add_command(label="Import DICOM", command=self.import_dicom)
        # Rest of the "File" menu
        filemenu.add_separator()
        filemenu.add_command(label="Save Segmentations", command=self.save_segmentations)
        filemenu.add_command(label="Exit", command=self.on_exit)


