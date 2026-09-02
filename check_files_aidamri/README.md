# Check AIDA MRI files

`check_files.py` recursively inventories `*.nii.gz` files and parses underscore-separated `sub-*`, `ses-*`, `run-*`, and modality components from their names.

```bash
pip install pandas
python check_files.py -i /path/to/project-or-proc_data
```

It prints the table and writes `processed_files_overview.csv` into the input directory. It does not validate NIfTI contents or full BIDS compliance.
