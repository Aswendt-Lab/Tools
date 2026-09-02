import os
import tkinter as tk
from tkinter import filedialog
import subprocess
import re

# Select directory
root = tk.Tk()
root.withdraw()
video_pathways = filedialog.askopenfilenames()

# Get cropping dimensions from user input
crop_width = int(input("Enter crop width in pixels: "))
crop_height = int(input("Enter crop height in pixels: "))
crop_x = int(input("Enter x-coordinate of top-left corner of crop area in pixels: "))
crop_y = int(input("Enter y-coordinate of top-left corner of crop area in pixels: "))

for video_pathway in video_pathways:
    # Extract video name without extension
    video_name = os.path.splitext(video_pathway)[0]

    # Get fps of video
    result = subprocess.run(['avprobe', '-show_streams', video_pathway], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stderr.decode('utf-8')
    duration = float(re.findall(r'duration=(\d+\.\d+)', output)[0])
    frames = int(re.findall(r'nb_frames=(\d+)', output)[0])
    fps = frames / duration

    # Create new video with adjusted fps and crop
    subprocess.run(['ffmpeg', '-i', video_pathway, '-filter:v', f"fps='{fps}'", f"{video_name}_fps.mp4"])
    subprocess.run(['ffmpeg', '-n', '-i', f"{video_name}_fps.mp4", '-filter:v',
                    f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}", '-map_metadata', '0', '-c:a', 'copy',
                    f"{video_name}_cropped.mp4"])

# Select directory
root = tk.Tk()
root.withdraw()
video_pathways = filedialog.askopenfilenames()

for video_pathway in video_pathways:
    # Extract video name without extension
    video_name = os.path.splitext(video_pathway)[0]

    # Get fps of video
    result = subprocess.run(['avprobe', '-show_streams', video_pathway], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stderr.decode('utf-8')
    duration = float(re.findall(r'duration=(\d+\.\d+)', output)[0])
    frames = int(re.findall(r'nb_frames=(\d+)', output)[0])
    fps = frames / duration

    # Create new video with adjusted fps and crop
    subprocess.run(['ffmpeg', '-i', video_pathway, '-filter:v', f"fps='{fps}'", f"{video_name}_fps.avi"])
    subprocess.run(['ffmpeg', '-n', '-i', f"{video_name}_fps.avi", '-filter:v',
                    f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}", '-map_metadata', '0', '-c:a', 'copy',
                    f"{video_name}_cropped.avi"])
