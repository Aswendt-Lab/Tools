import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox

def read_list(file_path):
    if not os.path.isfile(file_path):
        messagebox.showerror("Error", f"The path '{file_path}' is not a valid file.")
        return []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return list(set(lines))  # Return unique strings only

def copy_files(source_dir, destination_dir, search_strings):
    if not search_strings:
        messagebox.showwarning("Warning", "The list file is empty or invalid.")
        return

    print(f"Source Directory: {source_dir}")
    print(f"Destination Directory: {destination_dir}")
    print(f"Search Strings: {search_strings}")

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            print(f"Checking file: {file} in {root}")
            if any(s in file for s in search_strings):
                source_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, source_dir)
                destination_path = os.path.join(destination_dir, relative_path, file)
                print(f"Matched file: {source_file_path}")
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                try:
                    shutil.copy2(source_file_path, destination_path)
                    print(f'Copied: {source_file_path} to {destination_path}')
                except PermissionError as e:
                    print(f"Skipping file due to permission error: {source_file_path}. Error: {e}")
                except Exception as e:
                    print(f"Error copying file: {source_file_path}. Error: {e}")
    messagebox.showinfo("Success", "Files copied successfully.")

def select_source_directory():
    path = filedialog.askdirectory()
    if path:
        source_dir_var.set(path)

def select_destination_directory():
    path = filedialog.askdirectory()
    if path:
        destination_dir_var.set(path)

def select_list_file():
    path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if path:
        list_file_var.set(path)

def start_copying():
    source_dir = source_dir_var.get()
    destination_dir = destination_dir_var.get()
    list_file = list_file_var.get()
    if not source_dir or not destination_dir or not list_file:
        messagebox.showwarning("Warning", "Please select source directory, destination directory, and list file.")
        return
    search_strings = read_list(list_file)
    if not search_strings:
        messagebox.showerror("Error", "No valid strings found in the list file.")
        return
    copy_files(source_dir, destination_dir, search_strings)

# GUI setup
root = tk.Tk()
root.title("File Copier")

tk.Label(root, text="Source Directory:").grid(row=0, column=0, padx=10, pady=10)
source_dir_var = tk.StringVar()
tk.Entry(root, textvariable=source_dir_var, width=50).grid(row=0, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_source_directory).grid(row=0, column=2, padx=10, pady=10)

tk.Label(root, text="Destination Directory:").grid(row=1, column=0, padx=10, pady=10)
destination_dir_var = tk.StringVar()
tk.Entry(root, textvariable=destination_dir_var, width=50).grid(row=1, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_destination_directory).grid(row=1, column=2, padx=10, pady=10)

tk.Label(root, text="List File:").grid(row=2, column=0, padx=10, pady=10)
list_file_var = tk.StringVar()
tk.Entry(root, textvariable=list_file_var, width=50).grid(row=2, column=1, padx=10, pady=10)
tk.Button(root, text="Browse", command=select_list_file).grid(row=2, column=2, padx=10, pady=10)

tk.Button(root, text="Start Copying", command=start_copying).grid(row=3, column=0, columnspan=3, padx=10, pady=20)

root.mainloop()
