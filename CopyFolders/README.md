# CopyFolders

This folder contains three Tkinter GUI experiments:

- `Copy_Folder.py` copies matching **directories** using `rsync`.
- `Copy_Folder_v1.py` copies matching **files** with `shutil.copy2`.
- `Copy_Folder_v2.py` copies matching files and directories, skips ZIP files, skips same-size destination files, and displays per-file progress. This is the most capable version.

Each GUI expects a source directory, destination directory, and a text file containing one case-sensitive search string per line.

```bash
python Copy_Folder_v2.py
```

Matching is by name substring. Existing data can be merged or overwritten; test with a small directory first.
