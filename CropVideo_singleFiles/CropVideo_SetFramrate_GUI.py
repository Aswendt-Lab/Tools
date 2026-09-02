import os
import tkinter as tk
from tkinter import filedialog
import cv2

# Create root window
root = tk.Tk()
root.withdraw()

# Define allowed file types
filetypes = [("Video Files", (".mp4", ".avi"))]

# Ask user to select mode
mode = input("Please select mode (1 for single video mode, 2 for multiple videos mode): ")

if mode == "1":
    # Ask user to select a video file
    video_path = filedialog.askopenfilename(filetypes=filetypes)

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
    x_start, y_start, x_end, y_end = roi

    # Release video capture
    cap.release()

    # Define output path for cropped video
    output_path = os.path.join(video_dir, f"{video_name}_cropped{video_ext}")

    # Run ffmpeg command to crop video
    os.system(
        f"ffmpeg -i {video_path} -filter:v \"crop={x_end - x_start}:{y_end - y_start}:{x_start}:{y_start}\" -r {new_fps} {output_path}")

    print(f"{video_name}{video_ext} cropped and saved successfully.")

elif mode == "2":
    # Ask user to select a folder containing video files
    folder_path = filedialog.askdirectory()

    # Ask user to input desired FPS
    new_fps = input("Please enter a new FPS: ")

    # Ask user to select cropping area
    cap = cv2.VideoCapture(os.path.join(folder_path, os.listdir(folder_path)[0]))
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
    x_start, y_start, x_end, y_end = roi

    # Release video capture
    cap.release()

    # Loop through all video files in the folder and process them
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp4") or filename.endswith(".avi"):
            # Get video name and extension
            video_name, video_ext = os.path.splitext(filename)


