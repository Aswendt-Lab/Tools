# CropVideo4DLC

`CropVideo_SetFramrate_GUI_multiple.py` prepares MP4 videos for DeepLabCut. It lets you select multiple files, choose a crop rectangle from the first video, optionally set a new FPS, optionally rotate each cropped frame by 180°, and select an output directory. The same crop rectangle is applied to all selected videos.

## Requirements and usage

```bash
pip install opencv-python
python CropVideo_SetFramrate_GUI_multiple.py
```

Python must include Tkinter. Output names contain the FPS and `_crop`. The OpenCV `mp4v` encoder is used; audio is not copied. `cropvideo.yml` is an older environment specification and may need updating for the current platform.
