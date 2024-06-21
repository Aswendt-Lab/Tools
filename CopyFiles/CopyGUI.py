import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

def read_list(file_path):
    if not os.path.isfile(file_path):
        messagebox.showerror("Error", f"The path '{file_path}' is not a valid file.")
        return []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return list(set(lines))  # Return unique strings only

def copy_with_progress(source, destination, progress_bar):
    total_size = os.path.getsize(source)
    copied_size = 0

    with open(source, 'rb') as src, open(destination, 'wb') as dst:
        while True:
            buffer = src.read(1024 * 1024)  # Read in 1 MB chunks
            if not buffer:
                break
            dst.write(buffer)
            copied_size += len(buffer)
            progress = (copied_size / total_size) * 100
            progress_bar['value'] = progress
            root.update_idletasks()

def copy_files(source_dir, destination_dir, search_strings):
    if not search_strings:
        messagebox.showwarning("Warning", "The list file is empty or invalid.")
        return

    print(f"Source Directory: {source_dir}")
    print(f"Destination Directory: {destination_dir}")
    print(f"Search Strings: {search_strings}")

    progress_bar['value'] = 0
    for root, dirs, files in os.walk(source_dir):
        for name in dirs + files:
            if name.endswith('.zip'):
                print(f"Skipping zip file: {name}")
                continue

            if any(s in name for s in search_strings):
                source_path = os.path.join(root, name)
                relative_path = os.path.relpath(root, source_dir)
                destination_path = os.path.join(destination_dir, relative_path, name)
                print(f"Matched: {source_path}")

                if os.path.exists(destination_path):
                    if os.path.isfile(source_path) and os.path.getsize(source_path) == os.path.getsize(destination_path):
                        print(f"Skipping existing file with the same size: {destination_path}")
                        continue

                if os.path.isdir(source_path):
                    shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
                    print(f'Copied directory: {source_path} to {destination_path}')
                else:
                    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                    try:
                        copy_with_progress(source_path, destination_path, progress_bar)
                        print(f'Copied file: {source_path} to {destination_path}')
                    except PermissionError as e:
                        print(f"Skipping file due to permission error: {source_path}. Error: {e}")
                    except Exception as e:
                        print(f"Error copying file: {source_path}. Error: {e}")

    messagebox.showinfo("Success", "Files copied successfully.")
    progress_bar['value'] = 0

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

progress_bar = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
progress_bar.grid(row=3, column=0, columnspan=3, padx=10, pady=20)

tk.Button(root, text="Start Copying", command=start_copying).grid(row=4, column=0, columnspan=3, padx=10, pady=20)

root.mainloop()
