import os
import tkinter as tk
from tkinter import filedialog
import cv2
import subprocess

# Create root window
root = tk.Tk()
root.withdraw()

# Define allowed file types
filetypes = [("Video Files", (".mp4", ".avi"))]

# Ask user to select video files
video_pathways = filedialog.askopenfilenames(filetypes=filetypes)

for video_path in video_pathways:
    # Get video name and directory
    video_dir, video_name_ext = os.path.split(video_path)
    video_name, video_ext = os.path.splitext(video_name_ext)

    # Ask user to input desired FPS
    stream = cv2.VideoCapture(video_path)
    fps = stream.get(cv2.CAP_PROP_FPS)
    print(f"Current FPS of {video_name}{video_ext} is {fps}.")
    new_fps = input("Please enter a new FPS: ")
    stream.release()

    # Ask user to select cropping area
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) == ord(" "):
            cv2.destroyAllWindows()
            break

    # Get dimensions of selected cropping area
    roi = cv2.selectROI("Select ROI", frame)
    x_start, y_start, width, height = roi

    # Check that selected cropping area is within the dimensions of the original video
    stream = cv2.VideoCapture(video_path)
    width_original = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_original = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if x_start + width > width_original:
        width = width_original - x_start
    if y_start + height > height_original:
        height = height_original - y_start
    stream.release()

    # Check that starting coordinates are less than ending coordinates
    if x_start >= x_start + width or y_start >= y_start + height:
        print("Invalid cropping area selected. Please try again.")
        continue

    # Release video capture
    cap.release()

    # Define output path for cropped video
    output_path = os.path.join(video_dir, f"{video_name}_cropped{video_ext}")

    # Run ffmpeg command to crop video
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_path,
        "-filter:v", f"crop={width}:{height}:{x_start}:{y_start}",
        "-r", new_fps,
        "-y", output_path # overwrite output file if it already exists
    ]
    subprocess.run(ffmpeg_cmd, check=True)

print("All videos cropped and saved successfully.")
