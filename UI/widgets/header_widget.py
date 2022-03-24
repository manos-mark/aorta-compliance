import tkinter as tk
from .base_widget import BaseIOWidget

class HeaderWidget(BaseIOWidget):
    def __init__(self, parent, main_window):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.main_window = main_window
        
        # label header to be placed in the frame_header
        import_btn = tk.Button(self.parent, text="Import Dicom", command=self.import_dicom, bg='grey', fg='black', height='1', font=("Helvetica 16 bold"))
        # inside the grid of frame_header, place it in the position 0,0
        import_btn.grid(row=0, column=0, sticky='nsew')
        
        save_segmentations_btn = tk.Button(self.parent, text = "Save Segmentations", command=self.save_segmentations, bg='grey', fg='black', height='1', font=("Helvetica 16 bold"))
        # inside the grid of frame_header, place it in the position 0,0
        save_segmentations_btn.grid(row=0, column=1, sticky='nsew')
        
        exit_btn = tk.Button(self.parent, text = "Exit", command=self.on_exit, bg='grey', fg='black', height='1', font=("Helvetica 16 bold"))
        # inside the grid of frame_header, place it in the position 0,0
        exit_btn.grid(row=0, column=2, sticky='nsew')
        
        self.columnconfigure([0,1,2], weight=1)
        