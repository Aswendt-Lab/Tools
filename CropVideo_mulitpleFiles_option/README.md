# Multiple-video crop tools

This folder contains historical versions `CropVideo_SetFramrate_GUI_multiple_v1.py` through `v6.py`. Use `v6` unless reproducing an older workflow. It selects multiple MP4 files, uses the first video to define a crop box, optionally changes FPS and rotates frames by 180°, and writes cropped MP4 files to a chosen directory. Audio is not retained.

```bash
pip install opencv-python
python CropVideo_SetFramrate_GUI_multiple_v6.py
```

`Copy_videos.py` is a separate command-line utility that reads IDs from a file, searches a source tree for matching behavior or `iwxdata` files, records match lists, and copies selected files to a destination. Inspect its options with:

```bash
python Copy_videos.py --help
```

The `conda_env/cropvideo.yml` file describes an older environment. Local example videos are intentionally excluded from Git.
