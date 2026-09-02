# CorrectBIDS_AIDAmri

`mouse_bids_cleanup_all.py` performs project-specific cleanup on a mouse MRI BIDS tree. It:

- moves known helper files (`ScanProgram.scanProgram`, `subject`, `AdjStatePerStudy`, and `routine`) into `sourcedata/`;
- moves stray top-level `ses-*` directories under an inferred `sub-*` directory;
- renames EPI NIfTI/JSON pairs to BIDS `*_task-rest_bold` names;
- deletes field-map files smaller than 1 KB; and
- normalizes selected JSON metadata, including repetition time and malformed fields.

## Usage

```bash
python mouse_bids_cleanup_all.py /path/to/bids-root
```

Run `bids-validator` afterward. This script moves, renames, edits, and deletes files, so use a backup or version-controlled dataset and review the project-specific assumptions before running it.
