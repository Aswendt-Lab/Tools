# CropVideos label OCR

Despite the historical folder name, `CropVideoGUI.py` does not crop video. It samples frames, uses Tesseract OCR to detect label text, proposes a sanitized filename, and optionally renames videos in the selected top-level input directory.

```bash
pip install opencv-python pytesseract
brew install tesseract
python CropVideoGUI.py /path/to/videos --seconds 100 --sample-every 1
```

Add `--rename` to rename files in place. Without it, the script only prints proposed mappings. `--save-frames DIR` saves the first useful OCR frame. Existing target names receive a numeric suffix rather than being overwritten.
