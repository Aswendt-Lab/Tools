# IdentifyVideoFileName

Utilities that read visible experiment labels from video frames and create standardized filenames. The preferred implementation is `VideoIdentFileName.swift`, a recursive macOS command-line tool using Apple Vision OCR. The Python files are historical Tesseract/OpenCV variants for earlier label layouts.

## Current Swift tool

Requirements: macOS, Xcode Command Line Tools, and Swift.

```bash
xcode-select --install
swiftc -O VideoIdentFileName.swift -o VideoIdentFileName
./VideoIdentFileName /path/to/videos --dry-run --print-config
```

The tool scans `.mp4`, `.mov`, `.m4v`, `.avi`, and `.mkv` recursively. It copies a newly named file beside the original; it does not remove the source. Its target format is:

```text
<StudyID>_<Stage>_<Behavior>.<extension>
```

It infers behavior from folder names or accepts `--behavior NAME`. Already named videos are skipped unless `--include-named` is provided. Use `--dry-run` before allowing file operations.

Important options supported by the current source:

```text
--behavior NAME
--seconds N
--start-offset N
--step N
--retry-step N
--overwrite
--dry-run
--include-named
--max-p N
--subject-prefixes GV,SP,SR,PB,CC
--csv FILE
--debug-dir DIR
--debug-limit N
--print-config
--no-fullframe-fallback
```

`--mode` is accepted only for backward compatibility and its value is ignored. The current implementation does not support the formerly documented `--default-prefix` or `--out-dir` options.

`--overwrite` requests the exact target name, but the current implementation uses `copyItem` without first removing an existing destination. If that name already exists, the copy reports an error rather than replacing it; omit `--overwrite` to receive a safe numeric suffix.

## Historical Python tools

- `VideoIdentFileName_tesseract_recursive.py`: recursive yellow-label detection; renames in place unless `--dry-run` is used.
- `VideoIdentFileName_tesseract_labels.py`: scans only immediate input files and optionally renames with `--rename`.
- `VideoIdentFileName_rotating_beam.py`: orientation-robust Rotating Beam variant; copies by default to `renamed_copies/`, or renames with `--rename`.
- `VideoIdentFileName_cylinder.py`: yellow-card/Cylinder variant with optional `--behavior`; copies by default.

Python variants require OpenCV, NumPy, pytesseract, and a Tesseract installation. Inspect `--help` for the exact interface of a selected version. Example videos and compiled binaries are local-only and excluded from Git.
