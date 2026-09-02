# Single-file video crop tools

These are historical interactive crop/FPS utilities:

- `CropVideo_SetFramrate.py` selects files, determines FPS with `avprobe`, and uses `ffmpeg` to create intermediate `_fps` and final `_cropped` files.
- `CropVideo_SetFramrate_GUI_v1.py` and `v2.py` are earlier GUI variants retained for reproducibility.
- `CropVideo_SetFramrate_GUI.py` supports single-video or whole-folder processing and is the current version.

Requirements include Python with Tkinter, OpenCV, `ffmpeg`, and—specifically for `CropVideo_SetFramrate.py`—`avprobe`.

```bash
python CropVideo_SetFramrate_GUI.py
```

Some commands construct shell strings from file paths and may not handle spaces safely. Work on copies and prefer `CropVideo4DLC` for the maintained multi-file workflow.
