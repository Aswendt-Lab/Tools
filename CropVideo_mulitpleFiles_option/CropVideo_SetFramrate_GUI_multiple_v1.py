import os
import tkinter as tk
from tkinter import filedialog
import cv2

def get_videos():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title='Choose video file(s)', filetypes=[('Video Files', '*.mp4')])
    return file_paths

def get_crop_box(video_path):
    cap = cv2.VideoCapture(video_path)
    _, frame = cap.read()
    cap.release()
    return cv2.selectROI(frame)

def crop_video(video_path, crop_box):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name, ext = os.path.splitext(video_path)
    video_name += '_cropped'
    save_path = video_name + ext
    out = cv2.VideoWriter(save_path, fourcc, fps, (int(crop_box[2]), int(crop_box[3])))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[int(crop_box[1]):int(crop_box[1] + crop_box[3]), int(crop_box[0]):int(crop_box[0] + crop_box[2])]
        out.write(cropped_frame)
    cap.release()
    out.release()
    return save_path, fps

def process_multiple_videos():
    video_paths = get_videos()
    if not video_paths:
        print('No videos selected.')
        return
    crop_box = get_crop_box(video_paths[0])
    new_fps = input("Enter the new FPS for the videos (press Enter to keep original FPS): ")
    for video_path in video_paths:
        save_path, old_fps = crop_video(video_path, crop_box)
        if new_fps:
            old_fps_str = str(int(old_fps))
            print(f"{os.path.basename(video_path)} ({old_fps_str} FPS) cropped and saved with new FPS of {new_fps} as {os.path.basename(save_path)}.")
            change_fps(save_path, float(new_fps))
        else:
            print(f"{os.path.basename(video_path)} ({int(old_fps)} FPS) cropped and saved as {os.path.basename(save_path)}.")

def change_fps(video_path, new_fps):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name, ext = os.path.splitext(video_path)
    video_name += '_new_fps'
    save_path = video_name + ext
    out = cv2.VideoWriter(save_path, fourcc, new_fps, (width, height))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    os.replace(save_path, video_path)

process_multiple_videos()
