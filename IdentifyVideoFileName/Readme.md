Scripts to identify labels printed on yellow sticky notes for video recordings of cylinder, grid walk, and rotating beam test. Label is identified and file name updated with the correct id. Format: studyID_time point (e.g., GV_T3_3_1_Baseline).

For the swift version run first 

# VideoIdentFileName (Apple Vision OCR)

Rename experiment videos by reading text on **yellow sticky notes** directly from the frames (runs locally on macOS using Apple’s Vision OCR).  
Works well for patterns like:

- **Stage**: `Baseline` or `P1..P60`
- **Study ID**: e.g. `GV_T3_13_1`

> ⚠️ Paths contain spaces? **Quote them**. Multiline commands? Put a `\` at the **end of each line**.

---

## Build (macOS)

Requires Xcode Command Line Tools.

```bash
xcode-select --install   # if not installed
swiftc -O VideoIdentFileName_v6.swift -o VideoIdentFileName_v6
```
## Run (macOS)
```bash
./VideoIdentFileName_v6 "/path/with spaces/Group1_nolabels" \
  --mode subject_stage \
  --behavior Cylinder \
  --step 0.5 \
  --out-dir "/path/with spaces/Group1"
