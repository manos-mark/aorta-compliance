#!/usr/bin/env python

from tkinter import Tk
from mainwindow import MainWindow

# Main definition of the program
def main():
    root = Tk() # Create GUI object
    root.minsize(600, 400)  # Define min size of the window
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))    # Set size to the full screen size
    MainWindow(root)  # Assign all the functions in mainwindow to the app
    root.mainloop() # Start the main loop that initializes the app

if __name__ == '__main__':
    main()