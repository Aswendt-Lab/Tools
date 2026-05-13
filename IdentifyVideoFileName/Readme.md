# VideoIdentFileName

Utilities for renaming behavioral experiment videos by reading labels written on yellow sticky notes in the video frames.

The main tool is a **macOS Swift command-line app** using Apple Vision OCR. It is recommended for Apple Silicon Macs and for the current Gridwalk/Cylinder/Rotating Beam label format. Two older **Python/Tesseract fallback scripts** are also included for cases where the Swift OCR result is not satisfactory for a specific test setup.

Typical output filename:

```text
<StudyID>_<Stage>_<Behavior>.mp4
```

Examples:

```text
GV_T3_4_1_P8_Cylinder.mp4
SP_T1_11_2_P11_Gridwalk.mp4
GV_T3_3_1_Baseline_Gridwalk.mp4
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

## Recommended tool: Swift / Apple Vision OCR

Use this first on macOS. It is dependency-light, runs locally, and uses Apple’s Vision OCR. It is especially useful on Apple Silicon Macs.

### File name

Use the Swift source file as:

```text
VideoIdentFileName.swift
```

Compile it to a binary named:

```text
VideoIdentFileName
```

---

## Requirements

### macOS Swift version

Install Xcode Command Line Tools if needed:

```bash
xcode-select --install
```

Then compile:

```bash
swiftc -O VideoIdentFileName.swift -o VideoIdentFileName
```

No Python environment is required for the Swift version.

---

## Quick start: Swift version

### Gridwalk example

```bash
./VideoIdentFileName "/Volumes/Backup_AswendtLab/Archived_TVA_GFAP_Vimentin_Goeteborg/Behavior/Gridwalk/P11/Group1_nolabels" \
  --mode subject_stage \
  --behavior Gridwalk \
  --step 0.25 \
  --subject-prefixes "GV,SP,SR,PB,CC" \
  --out-dir "/Volumes/Backup_AswendtLab/Archived_TVA_GFAP_Vimentin_Goeteborg/Behavior/Gridwalk/P11/Group1" \
  --debug-dir "./debug_gridwalk" \
  --print-config
```

Expected output:

```text
SP_T1_11_2_P11_Gridwalk.mp4
```

### Cylinder example

```bash
./VideoIdentFileName "/Volumes/Backup_AswendtLab/Archived_TVA_GFAP_Vimentin_Goeteborg/Behavior/Cylinder/P7/Group2_nolabels" \
  --mode subject_stage \
  --behavior Cylinder \
  --step 0.5 \
  --subject-prefixes "GV,SP,SR,PB,CC" \
  --out-dir "/Volumes/Backup_AswendtLab/Archived_TVA_GFAP_Vimentin_Goeteborg/Behavior/Cylinder/P7/Group2"
```

Expected output:

```text
GV_T3_4_1_P8_Cylinder.mp4
```

### Rotating Beam example

```bash
./VideoIdentFileName "/path/to/RotatingBeam/P7/Group1_nolabels" \
  --mode subject_stage \
  --behavior RotatingBeam \
  --step 0.5 \
  --subject-prefixes "GV,SP,SR,PB,CC" \
  --out-dir "/path/to/RotatingBeam/P7/Group1"
```

Expected output:

```text
GV_T3_4_1_P8_RotatingBeam.mp4
```

---

## Command-line options for the Swift tool

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

### Options

#### `--mode <stage_behavior|subject_stage>`

Controls the naming mode.

Current recommended mode:

```bash
--mode subject_stage
```

The current Swift script always appends the behavior name, so both modes produce a behavior-tagged filename. When a study ID is detected, the output pattern is:

```text
<StudyID>_<Stage>_<Behavior>.mp4
```

Example:

```text
GV_T3_4_1_P8_Cylinder.mp4
```

If no study ID is detected, the fallback pattern is:

```text
<Stage>_<Behavior>.mp4
```

Example:

```text
P8_Cylinder.mp4
```

#### `--behavior <name>`

Behavior/test name to append to the filename.

Examples:

```bash
--behavior Gridwalk
--behavior Cylinder
--behavior RotatingBeam
```

#### `--seconds <N>`

Limit scanning to the first `N` seconds after `--start-offset`.

Example:

```bash
--seconds 120
```

If omitted, the tool scans the whole clip. Omitting `--seconds` is slower but more robust when labels appear late.

#### `--start-offset <N>`

Skip the first `N` seconds before scanning.

Example:

```bash
--start-offset 20
```

Default:

```text
0
```

#### `--step <N>`

Sampling interval in seconds.

Smaller values are slower but more thorough.

Recommended values:

```bash
--step 0.25   # thorough, recommended for small labels
--step 0.5    # good default
--step 1.0    # faster, less exhaustive
```

#### `--out-dir <folder>`

Copy renamed videos to this output folder.

Example:

```bash
--out-dir "/path/to/renamed/output"
```

If `--out-dir` is omitted, the tool renames files in place.

#### `--overwrite`

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

#### `--max-p <int>`

Maximum allowed `P` stage number.

Default:

```text
60
```

Example:

```bash
--max-p 80
```

#### `--subject-prefixes "GV,SP,SR,PB,CC"`

Comma-separated whitelist of accepted study-ID prefixes.

Default:

```bash
--subject-prefixes "GV,SP,SR,PB,CC"
```

Examples of accepted study IDs:

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

#### `--csv <path>`

Write an audit CSV file containing:

```text
source, stage, subject, new_name
```

Example:

```bash
--csv "./rename_log.csv"
```

#### `--debug-dir <folder>`

Save yellow sticky-note crops used for OCR. This is useful for checking why a file was misread or skipped.

Example:

```bash
--debug-dir "./debug_gridwalk"
```

#### `--print-config`

Print the parsed configuration before processing.

Useful for checking paths, behavior name, prefixes, and scan settings.

Example:

```bash
--print-config
```

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

## Python fallback scripts

The Swift tool is recommended. If it does not work well for a specific test, try the older Python/Tesseract scripts.

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

---

## Fallback: Rotating Beam script

File:

```text
VideoIdentFileName_v1.2_RB.py
```

Use this if the Swift version performs poorly on rotating beam recordings.

### Example: copy renamed videos to output folder

```bash
python VideoIdentFileName_v1.2_RB.py "/path/to/RotatingBeam/input" \
  --seconds 100 \
  --sample-every 1.0 \
  --out-dir "/path/to/RotatingBeam/renamed" \
  --copy
```

### Example: rename in place

```bash
python VideoIdentFileName_v1.2_RB.py "/path/to/RotatingBeam/input" \
  --seconds 100 \
  --sample-every 1.0 \
  --rename
```

### Options

```text
input_dir              Folder containing videos
--seconds              Seconds to scan from the start; default 100
--sample-every         Sample every N seconds; default 1.0
--save-frames          Save the first good frame to this folder
--extensions           Video extensions to include; default .mp4 .avi .mov .mkv .m4v
--rename               Rename originals in place; non-overwriting
--copy                 Copy files to --out-dir; default behavior
--out-dir              Output folder for copies; default <input>/renamed_copies
```

---

## Fallback: Cylinder script

File:

```text
VideoIdentFileName_v1.3_CY.py
```

Use this if the Swift version performs poorly on cylinder recordings.

### Example: copy renamed videos and append Cylinder

```bash
python VideoIdentFileName_v1.3_CY.py "/path/to/Cylinder/input" \
  --seconds 100 \
  --sample-every 1.0 \
  --behavior Cylinder \
  --out-dir "/path/to/Cylinder/renamed" \
  --copy
```

### Example: rename in place

```bash
python VideoIdentFileName_v1.3_CY.py "/path/to/Cylinder/input" \
  --seconds 100 \
  --sample-every 1.0 \
  --behavior Cylinder \
  --rename
```

### Options

```text
input_dir              Folder containing videos
--seconds              Seconds to scan from the start; default 100
--sample-every         Sample every N seconds; default 1.0
--save-frames          Save the first good frame to this folder
--extensions           Video extensions to include; default .mp4 .avi .mov .mkv .m4v
--behavior             Behavior/test name appended to the output filename, e.g. Cylinder
--rename               Rename originals in place; non-overwriting
--copy                 Copy files to --out-dir; default behavior
--out-dir              Output folder for copies; default <input>/renamed_copies
```

---

## Troubleshooting

### The study ID is missing

Try:

```bash
--step 0.25
--debug-dir "./debug_crops"
--subject-prefixes "GV,SP,SR,PB,CC"
```

Then inspect the PNG crops saved in `debug_crops`.

### The wrong prefix is detected

Check whether the prefix is included in:

```bash
--subject-prefixes "GV,SP,SR,PB,CC"
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

A file with the target name already exists in the output folder. This is intentional to avoid overwriting. Use `--overwrite` if replacement is desired.

---

## Recommended workflow

1. Run the Swift tool with `--out-dir` so original files are preserved.
2. Use `--debug-dir` for a small test batch.
3. Check the renamed files and debug crops.
4. Re-run on the full dataset.
5. Use the Python RB or CY fallback script only if the Swift OCR result is not acceptable.
