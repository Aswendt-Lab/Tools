import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import datetime

stop_copying = False  # Flag to stop the copying process

def read_list(file_path):
    if not os.path.isfile(file_path):
        messagebox.showerror("Error", f"The path '{file_path}' is not a valid file.")
        return []
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return list(set(lines))  # Return unique strings only

def copy_with_progress(source, destination, progress_bar, progress_label):
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
            progress_label.config(text=f"{progress:.2f}%")
            root.update_idletasks()

def copy_files(source_dir, destination_dir, search_strings):
    global stop_copying
    stop_copying = False

    if not search_strings:
        messagebox.showwarning("Warning", "The list file is empty or invalid.")
        return

    log_file_path = f"CopyGUI_{datetime.datetime.now().strftime('%Y-%m-%d')}.log"
    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Copy Log - {datetime.datetime.now()}\n\n")

        total_files = 0
        copied_files = 0
        failed_files = []

        progress_bar['value'] = 0
        progress_label.config(text="0%")

        for root_dir, dirs, files in os.walk(source_dir):
            if stop_copying:
                break

            for name in dirs + files:
                if stop_copying:
                    break

                print(f"Checking: {name}")
                if name.endswith('.zip'):
                    print(f"Skipping zip file: {name}")
                    continue

                if any(s in name for s in search_strings):
                    source_path = os.path.join(root_dir, name)
                    relative_path = os.path.relpath(root_dir, source_dir)
                    destination_path = os.path.join(destination_dir, relative_path, name)
                    total_files += 1

                    print(f"Matched: {source_path}")
                    if os.path.exists(destination_path):
                        if os.path.isfile(source_path) and os.path.getsize(source_path) == os.path.getsize(destination_path):
                            print(f"Skipping existing file with the same size: {destination_path}")
                            continue

                    try:
                        if os.path.isdir(source_path):
                            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
                            log_file.write(f"Copied directory: {source_path} to {destination_path}\n")
                        else:
                            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                            copy_with_progress(source_path, destination_path, progress_bar, progress_label)
                            log_file.write(f"Copied file: {source_path} to {destination_path}\n")
                        copied_files += 1
                    except Exception as e:
                        failed_files.append((source_path, str(e)))
                        log_file.write(f"Failed to copy {source_path}: {e}\n")

        log_file.write(f"\nTotal files found: {total_files}\n")
        log_file.write(f"Files successfully copied: {copied_files}\n")
        log_file.write(f"Files failed to copy: {len(failed_files)}\n")

        for failed_file, error in failed_files:
            log_file.write(f"{failed_file}: {error}\n")

    messagebox.showinfo("Success", f"Files copied successfully. Log saved to {log_file_path}.")
    progress_bar['value'] = 0
    progress_label.config(text="0%")
    root.destroy()  # Close the GUI

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

def stop_copying_process():
    global stop_copying
    stop_copying = True
    messagebox.showinfo("Process Stopped", "The copying process has been stopped.")

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

progress_label = tk.Label(root, text="0%")
progress_label.grid(row=4, column=0, columnspan=3)

tk.Button(root, text="Start Copying", command=start_copying).grid(row=5, column=0, padx=10, pady=20)
tk.Button(root, text="Stop Copying", command=stop_copying_process).grid(row=5, column=1, padx=10, pady=20)

root.mainloop()
