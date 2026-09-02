"""
Created on 07/08/2023

@author: Markus Aswendt + ChatGTP
AswendtLab: Neuroimaging & Neuroengineering
Department of Neurology
University Hospital Cologne, Germany
"""

import os
import subprocess
import tkinter as tk
from tkinter import filedialog

# Function to search for folders and perform the copying operation
def search_and_copy_files(search_directory, string_file, destination_directory):
    # Read the list of strings from the file
    with open(string_file, 'r') as file:
        search_strings = file.read().splitlines()

    for root, dirs, files in os.walk(search_directory):
        for name in dirs:
            if any(string.lower() in name.lower() for string in search_strings):
                source_path = os.path.join(root, name)
                destination_path = os.path.join(destination_directory, os.path.relpath(source_path, search_directory))

                # Create the destination directory for each folder copy
                os.makedirs(destination_path, exist_ok=True)

                # Copy the folder to the destination directory using rsync
                subprocess.run(['rsync', '-a', source_path + '/', destination_path])

# Function to handle the button click event
def process_files():
    # Get the selected directories and file
    search_directory = search_directory_var.get()
    string_file = string_file_var.get()
    destination_directory = destination_directory_var.get()

    # Call the function to search for folders and perform the copying operation
    search_and_copy_files(search_directory, string_file, destination_directory)

    # Show a message box indicating completion
    tk.messagebox.showinfo("Process Complete", "Folders have been copied!")

# Create the main window
window = tk.Tk()
window.title("Folder Search and Copy")

# Create variables to store the selected directory/file paths
search_directory_var = tk.StringVar()
string_file_var = tk.StringVar()
destination_directory_var = tk.StringVar()

# Function to handle the "Browse" button click event
def select_directory(var):
    directory = filedialog.askdirectory()
    var.set(directory)

def select_file(var):
    file = filedialog.askopenfilename()
    var.set(file)

# Create labels and entry fields for directory/file selection
tk.Label(window, text="Search Directory:").grid(row=0, column=0)
tk.Entry(window, textvariable=search_directory_var, width=40).grid(row=0, column=1)
tk.Button(window, text="Browse", command=lambda: select_directory(search_directory_var)).grid(row=0, column=2)

tk.Label(window, text="String File:").grid(row=1, column=0)
tk.Entry(window, textvariable=string_file_var, width=40).grid(row=1, column=1)
tk.Button(window, text="Browse", command=lambda: select_file(string_file_var)).grid(row=1, column=2)

tk.Label(window, text="Destination Directory:").grid(row=2, column=0)
tk.Entry(window, textvariable=destination_directory_var, width=40).grid(row=2, column=1)
tk.Button(window, text="Browse", command=lambda: select_directory(destination_directory_var)).grid(row=2, column=2)

# Create a button to initiate the folder search and copy process
tk.Button(window, text="Process Folders", command=process_files).grid(row=3, column=1)

# Start the main event loop
window.mainloop()
