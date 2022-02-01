import tkinter as tk


class StatusBarWidget(tk.Frame):
    height = 19

    def __init__(self, master): # Initialize the status bar and place it in the GUI
        tk.Frame.__init__(self, master)
        self.label = tk.Label(self, bd=1, relief='groove', anchor='w')
        self.label.pack(fill='both')

    def set(self, format_str, *args): # Change the text on the status bar
        self.label.config(text=format_str % args, font=("TkDefaultFont",12))
        self.label.update_idletasks()

    def clear(self): # Clear the text
        self.label.config(text="")
        self.label.update_idletasks()