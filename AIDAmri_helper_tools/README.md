# AIDAmri helper tools

`find_corrupted_method_files.py` recursively checks Bruker-style acquisition folders. It reports directories containing an `*acqp` file but no sibling `method` file, and directories whose `subject` file cannot be read or lacks a non-empty `##OWNER=` entry.

```bash
pip install pandas
python find_corrupted_method_files.py -i /path/to/input -o problems.csv
```

The CSV contains `Path_Without_Method` and `Corrupted_Subject_Path` columns. The script only reads source data.
