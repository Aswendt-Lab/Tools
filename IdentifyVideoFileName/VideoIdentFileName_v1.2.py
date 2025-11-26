""""
Created on 15.11.2025

@authors: Markus Aswendt, ChatGPT
Department of Neurology
University Hospital Frankfurt
Theodor-Stern-Kai 7
D-60590 Frankfurt am Main

"""

#!/usr/bin/env python3
"""
Scan videos, OCR the first ~100s to find labels (e.g., GV_T3_7_3 + Baseline),
and build a new name like GV_T3_7_3_Baseline.

Default: COPY each source video to <input>/renamed_copies (or --out-dir)
with the OCR-derived name, never overwriting (adds _1, _2, ... when needed).
Use --rename to rename in place (also non-overwriting).

Examples
--------
python label_namer.py /path/to/videos
python label_namer.py /path/to/videos --out-dir /path/to/output
python label_namer.py /path/to/videos --rename
python label_namer.py /path/to/videos --save-frames ./frames
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytesseract

# ---------- OCR helpers (orientation-aware) ----------

def rotate_bound(image, angle_degrees: float):
    """Rotate image without cropping (like imutils.rotate_bound, but no dependency)."""
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_degrees, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])

    # compute new bounding dimensions
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_for_ocr(img_bgr):
    """Strong, general-purpose preprocessing for printed labels."""
    # upscale a bit to help small text
    img = cv2.resize(img_bgr, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # local contrast
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # gentle denoise
    gray = cv2.fastNlMeansDenoising(gray, h=8)

    # adaptive threshold (works well for uneven lighting)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    return thr

def _tess_tokens(bin_img):
    """Run Tesseract and return tokens with confidences."""
    cfg = "--oem 3 --psm 6 -l eng"
    data = pytesseract.image_to_data(bin_img, config=cfg, output_type=pytesseract.Output.DICT)
    tokens = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        if not txt:
            continue
        try:
            conf = float(conf)
        except Exception:
            conf = -1.0
        if conf >= 55:  # a touch lower since angled text can be trickier
            tokens.append((txt.strip(), conf))
    return tokens

def ocr_text_orientation_robust(img_bgr):
    """
    Try multiple rotations to handle tilted/upside-down labels.
    Returns a merged list of (text, conf) tokens from the best rotations.
    """
    # Candidate angles: 0, ±10, ±20 AND 180, 180±10, 180±20
    angles = [0, -20, -10, 10, 20, 180, 160, 170, 190, 200]

    best_sets = []
    for ang in angles:
        rotated = rotate_bound(img_bgr, ang) if ang != 0 else img_bgr
        pre = preprocess_for_ocr(rotated)
        tokens = _tess_tokens(pre)
        if tokens:
            # score by total confidence to keep the best few rotations
            score = sum(c for _, c in tokens)
            best_sets.append((score, tokens))

    # Merge top-2 results (helps when one card is upright and the other inverted)
    best_sets.sort(key=lambda x: x[0], reverse=True)
    merged = []
    for _, toks in best_sets[:2]:
        merged.extend(toks)
    return merged

ID_REGEX = re.compile(r"\b[A-Z]{1,4}(?:_[A-Z0-9]+){1,6}\b")
WORD_REGEX = re.compile(r"\b[A-Za-z][A-Za-z\-]+\b")

def parse_labels(tokens):
    """
    Guess:
      - id_code: e.g., 'GV_T3_7_3' / 'SP_T1_3_1'
      - condition: e.g., 'Baseline'
    """
    texts = [t[0] for t in tokens]
    joined = " ".join(texts)

    ids = ID_REGEX.findall(joined)

    words = [w for w in WORD_REGEX.findall(joined)
             if len(w) >= 4 and not ID_REGEX.fullmatch(w)]
    priority = ["Baseline", "Pre", "Post", "Control", "Day1", "Day2", "Day3"]
    cond = next((p for p in priority if any(p.lower() == w.lower() for w in words)), None)
    if not cond and words:
        cond = Counter(w.lower() for w in words).most_common(1)[0][0].capitalize()

    id_code = Counter(ids).most_common(1)[0][0] if ids else None
    return id_code, cond

def sanitize(s: str) -> str:
    """Keep only A–Z, 0–9, and underscores; convert spaces to underscores."""
    return re.sub(r"[^A-Za-z0-9_]+", "", s.replace(" ", "_"))

# ---------- Video processing ----------

def scan_video(path: Path, seconds=100, sample_every=1.0, save_frame_dir: Path | None=None):
    """
    Scan the first `seconds` of a video, sampling every `sample_every` seconds.
    Returns (new_base_name or None, first_good_frame_path or None)
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {path}", file=sys.stderr)
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total_frames / fps if fps > 0 else seconds
    max_seconds = min(seconds, int(duration))

    id_counts = Counter()
    cond_counts = Counter()
    first_saved = None

    # Try left, right, and full width (labels often sit on sides)
    crops = [(0.00, 0.33), (0.66, 1.00), (0.0, 1.0)]

    step = max(1, int(sample_every))
    for sec in range(0, max_seconds + 1, step):
        if fps <= 0:
            break
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        tokens_all = []
        for x0f, x1f in crops:
            x0 = int(w * x0f)
            x1 = int(w * x1f)
            crop = frame[:, x0:x1]
            tokens_all.extend(ocr_text_orientation_robust(crop))

        id_code, cond = parse_labels(tokens_all)
        if id_code:
            id_counts[id_code] += 1
        if cond:
            cond_counts[cond] += 1

        if id_code and cond and save_frame_dir and not first_saved:
            save_frame_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_frame_dir / f"{sanitize(id_code)}_{sanitize(cond)}_frame{frame_idx}.jpg"
            cv2.imwrite(str(out_path), frame)
            first_saved = out_path

    cap.release()

    if not id_counts and not cond_counts:
        return None, first_saved

    id_best = id_counts.most_common(1)[0][0] if id_counts else None
    cond_best = cond_counts.most_common(1)[0][0] if cond_counts else None

    if id_best and cond_best:
        new_name = f"{sanitize(id_best)}_{sanitize(cond_best)}"
    elif id_best:
        new_name = sanitize(id_best)
    elif cond_best:
        new_name = sanitize(cond_best)
    else:
        new_name = None

    return new_name, first_saved

def unique_target(out_dir: Path, base: str, ext: str) -> Path:
    """Return a non-existing path like base.ext, base_1.ext, base_2.ext, ..."""
    target = out_dir / f"{base}{ext}"
    if not target.exists():
        return target
    i = 1
    while True:
        candidate = out_dir / f"{base}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1

# ---------- Main CLI (unchanged interface) ----------

def main():
    ap = argparse.ArgumentParser(description="Extract labels from videos and create safe new names (orientation-robust).")
    ap.add_argument("input_dir", type=Path, help="Folder containing videos")
    ap.add_argument("--seconds", type=int, default=100, help="How many seconds to scan from the start (default: 100)")
    ap.add_argument("--sample-every", dest="sample_every", type=float, default=1.0,
                    help="Sample every N seconds (default: 1.0)")
    ap.add_argument("--save-frames", dest="save_frames", type=Path, default=None,
                    help="Folder to save the first good frame (optional)")
    ap.add_argument("--extensions", nargs="+",
                    default=[".mp4", ".avi", ".mov", ".mkv", ".m4v"],
                    help="Video file extensions to include")

    # Mode selection: default is COPY; you can opt-in to renaming in place.
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--rename", action="store_true",
                      help="Rename originals in place (non-destructive; adds suffix if needed).")
    mode.add_argument("--copy", action="store_true",
                      help="Copy to --out-dir (default behavior).")

    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output folder for copies (default: <input>/renamed_copies)")

    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input folder does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    do_copy = args.copy or not args.rename  # default to copy
    out_dir = args.out_dir or (args.input_dir / "renamed_copies")
    if do_copy:
        out_dir.mkdir(parents=True, exist_ok=True)

    mappings = []
    for p in sorted(args.input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in [e.lower() for e in args.extensions]:
            continue

        print(f"[INFO] Processing: {p.name}")
        new_base, saved = scan_video(
            p,
            seconds=args.seconds,
            sample_every=args.sample_every,
            save_frame_dir=args.save_frames,
        )

        if new_base:
            if do_copy:
                target = unique_target(out_dir, new_base, p.suffix)
                shutil.copy2(p, target)
                print(f"  -> Copied to: {target.name}")
                mappings.append((p.name, target.name))
            else:
                target = unique_target(p.parent, new_base, p.suffix)
                p.rename(target)
                print(f"  -> Renamed to: {target.name}")
                mappings.append((p.name, target.name))
        else:
            print("  -> No reliable labels detected.")
            mappings.append((p.name, None))

        if saved:
            print(f"  -> Saved frame: {saved}")

    # Summary
    print("\n=== Summary ===")
    for orig, new in mappings:
        print(f"{orig}  =>  {new if new else '(no match)'}")

if __name__ == "__main__":
    main()
