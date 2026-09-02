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


def crop_video(video_path, save_dir, crop_box, new_fps=None, mirror=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not new_fps:
        new_fps = fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    head, tail = os.path.split(video_path)
    video_name, ext = os.path.splitext(tail)

    video_name += '_' + str(new_fps)  # adds fps into file name
    video_name += '_crop'
    save_name = video_name + ext
    save_path = os.path.join(save_dir, save_name)

    out = cv2.VideoWriter(save_path, fourcc, new_fps, (int(crop_box[2]), int(crop_box[3])))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped_frame = frame[int(crop_box[1]):int(crop_box[1] + crop_box[3]), int(crop_box[0]):int(crop_box[0] + crop_box[2])]
        if mirror:
            cropped_frame = cv2.flip(cropped_frame, 1)  # Horizontal flip
            cropped_frame = cv2.flip(cropped_frame, 0)  # Vertical flip
        out.write(cropped_frame)

    cap.release()
    out.release()
    return save_path, new_fps


def process_multiple_videos():
    video_paths = get_videos()
    if not video_paths:
        print('No videos selected.')
        return
    print('Videos found:')
    for video_path in video_paths:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"{os.path.basename(video_path)} ({fps} FPS)")

    new_fps = input("Enter the new FPS for the videos (press Enter to keep original FPS): ")
    if new_fps:
        new_fps = float(new_fps)

    crop_box = get_crop_box(video_paths[0])

    mirror_video = input("Do you want to mirror the video(s)? (yes or no): ")
    mirror = mirror_video.lower() == 'yes'

    save_dir = input("Where do you want to save the cropped videos? ")
    save_dir = save_dir.strip()

    for video_path in video_paths:
        save_path, fps = crop_video(video_path, save_dir, crop_box, new_fps, mirror)
        if new_fps:
            old_fps_str = str(int(fps))
            print(f"{os.path.basename(video_path)} ({old_fps_str} FPS) cropped and saved with new FPS of {new_fps} as {os.path.basename(save_path)}.")
        else:
            print(f"{os.path.basename(video_path)} ({int(fps)} FPS) cropped and saved as {os.path.basename(save_path)}.")


process_multiple_videos()
