# VideoIdentFileName

Utilities for renaming behavioral experiment videos by reading visible labels in video frames.

The main tool is a **macOS Swift command-line app** using Apple Vision OCR. It is recommended for Apple Silicon Macs and for the current Gridwalk/Cylinder/Rotating Beam label formats.

The repository should contain the Swift source file as:

```text
VideoIdentFileName.swift
```

Do not use versioned filenames in the repository. Compile this source file into the executable:

```text
VideoIdentFileName
```

Typical output filename:

```text
<StudyID>_<Stage>_<Behavior>.mp4
```

Examples:

```text
GV_T3_4_1_P8_Cylinder.mp4
SP_T1_11_2_P11_Gridwalk.mp4
GV_T3_3_1_Baseline_Gridwalk.mp4
GV_T3_73_7_Baseline_Cylinder.mp4
```

The current supported study-ID prefixes are:

```text
GV, SP, SR, PB, CC
```

The stage/timepoint is expected to be one of:

```text
Baseline, P1, P2, ..., P60
```

---

## What the tool detects

The Swift tool reads labels directly from sampled video frames using Apple Vision OCR.

It supports:

- yellow sticky-note labels
- white or light paper labels
- printed labels
- handwritten labels, when OCR quality is sufficient
- full study IDs such as `GV_T3_73_7`
- prefix-less handwritten IDs such as `T3_73_7`, which are converted to `GV_T3_73_7` by default

The default prefix for prefix-less IDs is:

```text
GV
```

You can change it with:

```bash
--default-prefix SP
```

---

## Filename validation

A video is considered correctly named only if it follows the full format:

```text
<StudyID>_<Stage>_<Behavior>.mp4
```

For example, this is valid:

```text
GV_T3_73_7_Baseline_Cylinder.mp4
```

This is **not** considered fully named, because the StudyID is missing:

```text
Baseline_Cylinder.mp4
Baseline_Cylinder_01.mp4
```

Such files will be processed and renamed if a StudyID can be detected from the video label.

If the filename already contains a valid StudyID, that StudyID is protected and treated as ground truth. OCR is not allowed to silently replace it with a different animal ID.

Example:

```text
GV_T3_12_1_Baseline_Cylinder 1.mp4
```

If OCR incorrectly detects `GV_T3_129`, the script keeps the filename ID `GV_T3_12_1` and ignores the conflicting OCR result.

---

## Requirements

Install Xcode Command Line Tools if needed:

```bash
xcode-select --install
```

Compile:

```bash
swiftc -O VideoIdentFileName.swift -o VideoIdentFileName
```

No Python environment is required for the Swift version.

---

## Quick start

### Cylinder example

```bash
./VideoIdentFileName "/Volumes/Projects/02_Archived_TVA_GFAP_Vimentin_Goeteborg/Behavior/Cylinder" \
  --mode subject_stage \
  --behavior Cylinder \
  --step 0.5 \
  --seconds 120 \
  --subject-prefixes "GV,SP,SR,PB,CC" \
  --default-prefix GV \
  --debug-dir "./debug_cylinder" \
  --print-config
```

Expected output example:

```text
GV_T3_73_7_Baseline_Cylinder.mp4
```

### Gridwalk example

```bash
./VideoIdentFileName "/Volumes/Projects/02_Archived_TVA_GFAP_Vimentin_Goeteborg/Behavior/Gridwalk/P11/Group1_nolabels" \
  --mode subject_stage \
  --behavior Gridwalk \
  --step 0.25 \
  --seconds 120 \
  --subject-prefixes "GV,SP,SR,PB,CC" \
  --out-dir "/Volumes/Projects/02_Archived_TVA_GFAP_Vimentin_Goeteborg/Behavior/Gridwalk/P11/Group1" \
  --debug-dir "./debug_gridwalk" \
  --print-config
```

Expected output example:

```text
SP_T1_11_2_P11_Gridwalk.mp4
```

### Rotating Beam example

```bash
./VideoIdentFileName "/path/to/RotatingBeam/P7/Group1_nolabels" \
  --mode subject_stage \
  --behavior RotatingBeam \
  --step 0.5 \
  --seconds 120 \
  --subject-prefixes "GV,SP,SR,PB,CC" \
  --out-dir "/path/to/RotatingBeam/P7/Group1"
```

Expected output example:

```text
GV_T3_4_1_P8_RotatingBeam.mp4
```

---

## Command-line options

The Swift script prints usage information when run without arguments:

```bash
./VideoIdentFileName
```

If your local copy implements `-h` or `--help`, you can also use:

```bash
./VideoIdentFileName -h
./VideoIdentFileName --help
```

### Required argument

```text
<root>
```

Root folder to scan. The tool recursively searches for video files inside this folder.

Supported video extensions:

```text
.mp4, .mov, .m4v, .avi, .mkv
```

### `--mode <stage_behavior|subject_stage>`

Controls the naming mode.

Recommended mode:

```bash
--mode subject_stage
```

When a StudyID is detected, the output pattern is:

```text
<StudyID>_<Stage>_<Behavior>.mp4
```

Example:

```text
GV_T3_4_1_P8_Cylinder.mp4
```

### `--behavior <name>`

Behavior/test name to append to the filename.

Examples:

```bash
--behavior Gridwalk
--behavior Cylinder
--behavior RotatingBeam
```

### `--seconds <N>`

Limit scanning to the first `N` seconds after `--start-offset`.

Example:

```bash
--seconds 120
```

If omitted, the tool scans the whole clip. Omitting `--seconds` is slower but more robust when labels appear late.

### `--start-offset <N>`

Skip the first `N` seconds before scanning.

Example:

```bash
--start-offset 20
```

Default:

```text
0
```

### `--step <N>`

Sampling interval in seconds.

Smaller values are slower but more thorough.

Recommended values:

```bash
--step 0.25   # thorough, recommended for small or handwritten labels
--step 0.5    # good default
--step 1.0    # faster, less exhaustive
```

### `--out-dir <folder>`

Copy renamed videos to this output folder.

Example:

```bash
--out-dir "/path/to/renamed/output"
```

If `--out-dir` is omitted, the tool renames files in place.

### `--overwrite`

Allow overwriting existing target files.

By default, the tool never overwrites files. If a target filename already exists, it adds a suffix:

```text
GV_T3_4_1_P8_Cylinder.mp4
GV_T3_4_1_P8_Cylinder_01.mp4
GV_T3_4_1_P8_Cylinder_02.mp4
```

Use with care:

```bash
--overwrite
```

### `--max-p <int>`

Maximum allowed `P` stage number.

Default:

```text
60
```

Example:

```bash
--max-p 80
```

### `--subject-prefixes "GV,SP,SR,PB,CC"`

Comma-separated whitelist of accepted StudyID prefixes.

Default:

```bash
--subject-prefixes "GV,SP,SR,PB,CC"
```

Examples of accepted StudyIDs:

```text
GV_T3_4_1
SP_T1_11_2
SR_T2_3_1
PB_T4_5_2
CC_T1_7_3
```

If you need to temporarily disable prefix filtering, use an empty string:

```bash
--subject-prefixes ""
```

### `--default-prefix <prefix>`

Prefix used when the video label contains a prefix-less StudyID.

Example OCR label:

```text
T3_73_7
```

With the default prefix, this becomes:

```text
GV_T3_73_7
```

Default:

```bash
--default-prefix GV
```

Example for another study prefix:

```bash
--default-prefix SP
```

### `--csv <path>`

Write an audit CSV file.

Example:

```bash
--csv "./rename_log.csv"
```

Depending on the script version, the CSV may include fields such as:

```text
source, stage, subject, new_name, filename_subject, ocr_subject, final_subject, action
```

Use the CSV to review which StudyID came from the original filename and which StudyID came from OCR.

### `--debug-dir <folder>`

Save OCR crops for debugging.

This now includes crops from yellow labels, white/light labels, and fallback frame regions.

Example:

```bash
--debug-dir "./debug_crops"
```

Inspect these crops when a file is skipped with:

```text
[SKIP] No valid StudyID
```

### `--debug-limit <N>`

Maximum number of debug crops to save.

Example:

```bash
--debug-limit 100
```

### `--print-config`

Print the parsed configuration before processing.

Useful for checking paths, behavior name, prefixes, default prefix, and scan settings.

Example:

```bash
--print-config
```

---

## Output behavior

### Already correctly named files

Files that already follow the complete format are skipped or marked as already named:

```text
GV_T3_73_7_Baseline_Cylinder.mp4
```

### Incomplete names

Files without a StudyID are processed:

```text
Baseline_Cylinder.mp4
Baseline_Cylinder_01.mp4
```

If OCR finds `T3_73_7` on the label, the output becomes:

```text
GV_T3_73_7_Baseline_Cylinder.mp4
```

### Conflicting StudyIDs

If the filename contains one StudyID and OCR detects another, the script protects the filename StudyID.

Example:

```text
Filename StudyID: GV_T3_12_1
OCR StudyID:      GV_T3_129
```

The filename StudyID is used, and the OCR conflict is ignored or logged as a warning.

---

## Important shell tips

### Quote paths

Always quote paths that may contain spaces:

```bash
"/Volumes/My Drive/Behavior/Gridwalk/P11/Group1_nolabels"
```

### Multiline commands

When splitting a command across multiple lines, put a backslash at the end of every continued line:

```bash
./VideoIdentFileName "/path/to/input" \
  --mode subject_stage \
  --behavior Gridwalk \
  --step 0.25 \
  --out-dir "/path/to/output"
```

There must be nothing after the backslash, not even a space.

### Avoid smart dashes

Use two normal ASCII hyphens:

```bash
--behavior Gridwalk
```

Not a copied smart dash:

```text
—behavior Gridwalk
```

The Swift script normalizes common Unicode dashes, but using ASCII `--` is still recommended.

---

## Troubleshooting

### A file is skipped with `No valid StudyID`

Example:

```text
[SKIP] No valid StudyID: .../Baseline_Cylinder.mp4
```

This means the script detected or inferred the stage and behavior but could not extract a valid StudyID from either the filename or OCR.

Try:

```bash
--step 0.25
--seconds 120
--debug-dir "./debug_crops"
--debug-limit 100
--subject-prefixes "GV,SP,SR,PB,CC"
--default-prefix GV
```

Then inspect the PNG crops saved in `debug_crops`.

For white handwritten labels, make sure the label is visible in at least some saved crops. If it is not visible, increase the scanned time window or omit `--seconds`.

### White handwritten labels are missed

Use a smaller step size and save debug crops:

```bash
./VideoIdentFileName "/path/to/input" \
  --behavior Cylinder \
  --step 0.25 \
  --seconds 120 \
  --debug-dir "./debug_crops" \
  --debug-limit 100
```

The current Swift tool searches yellow labels, white/light labels, and fallback regions, but OCR can still fail when handwriting is very faint, blurred, partly outside the frame, or strongly tilted.

### A prefix-less StudyID is detected

If the label says:

```text
T3_73_7
```

The script converts it to:

```text
GV_T3_73_7
```

To use another prefix:

```bash
--default-prefix SP
```

### The wrong prefix is detected

Check whether the prefix is included in:

```bash
--subject-prefixes "GV,SP,SR,PB,CC"
```

For prefix-less labels, also check:

```bash
--default-prefix GV
```

### Labels appear late in the video

Omit `--seconds` so the full clip is scanned:

```bash
./VideoIdentFileName "/path/to/input" --behavior Gridwalk --step 0.5
```

### Processing is too slow

Increase the frame sampling interval:

```bash
--step 1.0
```

### A file gets `_01` appended

A file with the target name already exists in the output folder. This is intentional to avoid overwriting.

Use `--overwrite` only if replacement is desired:

```bash
--overwrite
```

---

## Recommended workflow

1. Compile `VideoIdentFileName.swift` to `VideoIdentFileName`.
2. Run the Swift tool on a small test folder first.
3. Use `--debug-dir` and inspect the saved crops, especially for white handwritten labels.
4. Use `--out-dir` for the first full run if you want to preserve the original files.
5. Review the renamed files and optional CSV audit log.
6. Re-run on the full dataset once the settings are correct.

---

## Python fallback scripts

The Swift tool is recommended. Older Python/Tesseract fallback scripts may be kept in the repository for legacy use, but the main GitHub workflow should use:

```text
VideoIdentFileName.swift
```

### Python requirements

Create a conda environment:

```bash
conda create -n video-label-ocr -c conda-forge python=3.10 opencv numpy pytesseract tesseract ffmpeg -y
conda activate video-label-ocr
```

Verify:

```bash
python -c "import cv2, pytesseract, numpy; print('OK')"
tesseract --version
```

### Fallback: Rotating Beam script

File:

```text
VideoIdentFileName_RB.py
```

Example:

```bash
python VideoIdentFileName_RB.py "/path/to/RotatingBeam/input" \
  --seconds 100 \
  --sample-every 1.0 \
  --out-dir "/path/to/RotatingBeam/renamed" \
  --copy
```

### Fallback: Cylinder script

File:

```text
VideoIdentFileName_CY.py
```

Example:

```bash
python VideoIdentFileName_CY.py "/path/to/Cylinder/input" \
  --seconds 100 \
  --sample-every 1.0 \
  --behavior Cylinder \
  --out-dir "/path/to/Cylinder/renamed" \
  --copy
```
