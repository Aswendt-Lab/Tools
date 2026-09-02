# CopyFiles

`CopyGUI.py` is a Tkinter GUI that recursively searches a source directory for files or folders whose names contain any string listed in a text file. Matches are copied to a destination while preserving their relative directory structure.

ZIP files are skipped. An existing destination file with the same size is skipped; other matching files may be overwritten. A dated `CopyGUI_YYYY-MM-DD.log` is written in the current working directory.

## Usage

Create a UTF-8 text file containing one search string per line, then run:

```bash
python CopyGUI.py
```

Select the source directory, destination directory, and list file in the GUI. Python 3 with Tkinter is required.

![GUI example](https://github.com/Aswendt-Lab/Tools/assets/32373094/cb9e123e-ee99-4306-a240-09beadba6411)
