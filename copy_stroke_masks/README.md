# Copy stroke masks

`copy_stroke_masks.py` searches a configured source tree for `*Stroke_mask*`, matches the inferred subject name to `sub-*` folders in a configured processed-data tree, and **moves** each matching mask into the first available `anat` directory among `ses-P1`, `ses-P2`, `ses-P3`, and `ses-P7`. It writes `move_status.csv` in the source directory.

Before running, edit the hard-coded Windows paths `input_path` and `output_path` near the top of the script:

```bash
python copy_stroke_masks.py
```

This script moves rather than copies files and uses the first matching mask/session. Back up the data and review matches before use.
