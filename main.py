from tkinter import Tk
from widgets.core_widgets import MainWindow

# Main definition of the program
def main():
    window = Tk() # Create GUI object
    window.minsize(600, 400)  # Define min size of the window
    window.geometry("{0}x{1}+0+0".format(window.winfo_screenwidth(), window.winfo_screenheight()))    # Set size to the full screen size
    MainWindow(window)  # Assign all the functions in mainwindow to the app
    window.mainloop() # Start the main loop that initializes the app

if __name__ == '__main__':
    main()