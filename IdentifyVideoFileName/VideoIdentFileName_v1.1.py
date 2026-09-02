""""
Created on 05.11.2025

@authors: Markus Aswendt, ChatGPT
Department of Neurology
University Hospital Frankfurt
Theodor-Stern-Kai 7
D-60590 Frankfurt am Main

"""

#!/usr/bin/env python3
"""
Scan videos, read labels in the first 100 seconds, and build a new file name.

Examples
--------
python label_renamer.py /path/to/videos
python label_renamer.py /path/to/videos --rename
python label_renamer.py /path/to/videos --save-frames ./frames
"""

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pytesseract

# ---------- OCR helpers ----------

def ocr_text(img):
    """Run OCR on a BGR image and return a list of (text, conf) tokens."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Slight denoise + adaptive threshold to help printed labels
    blur = cv2.medianBlur(gray, 3)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    cfg = "--oem 3 --psm 6 -l eng"
    data = pytesseract.image_to_data(thr, config=cfg, output_type=pytesseract.Output.DICT)
    tokens = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        if not txt:
            continue
        try:
            conf = float(conf)
        except Exception:
            conf = -1.0
        if conf >= 60:  # keep reasonably confident tokens
            tokens.append((txt.strip(), conf))
    return tokens

ID_REGEX = re.compile(r"\b[A-Z]{1,4}(?:_[A-Z0-9]+){1,6}\b")
WORD_REGEX = re.compile(r"\b[A-Za-z][A-Za-z\-]+\b")

def parse_labels(tokens):
    """
    From OCR tokens, guess:
    - right/ID-like code: e.g., 'GV_T3_7_3' or 'SP_T1_3_1'
    - left/condition word: e.g., 'Baseline'
    Returns (id_code or None, condition or None).
    """
    texts = [t[0] for t in tokens]
    joined = " ".join(texts)

    # Find all ID-like patterns
    ids = ID_REGEX.findall(joined)

    # Find candidate condition words, prefer common words seen on yellow cards
    words = WORD_REGEX.findall(joined)
    # Exclude words that are part of the ID or very short
    words = [w for w in words if len(w) >= 4 and not ID_REGEX.fullmatch(w)]
    # Preference ordering
    priority = ["Baseline", "Pre", "Post", "Control", "Day1", "Day2", "Day3"]
    cond = None
    for p in priority:
        if any(p.lower() == w.lower() for w in words):
            cond = p
            break
    if not cond and words:
        # fallback: most frequent alphabetical word (case-insensitive)
        counts = Counter(w.lower() for w in words)
        cond = max(counts.items(), key=lambda x: x[1])[0].capitalize()

    id_code = None
    if ids:
        # choose the longest (usually the most complete) and most frequent
        counts = Counter(ids)
        id_code = max(counts.items(), key=lambda x: (x[1], len(x[0])))[0]

    return id_code, cond

def sanitize(s):
    """Keep only A–Z, 0–9, and underscores; convert spaces to underscores."""
    s = s.replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9_]+", "", s)

# ---------- Video processing ----------

def scan_video(path, seconds=100, sample_every=1.0, save_frame_dir=None):
    """
    Scan the first `seconds` of a video, sampling every `sample_every` seconds.
    Returns:
        new_name (str or None),
        first_good_frame_path (Path or None)
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

    # To speed up OCR, also try the left/right thirds where labels often sit
    crops = [(0.00, 0.33), (0.33, 0.66), (0.66, 1.00), (0.0, 1.0)]  # x-range fractions

    for sec in range(0, max_seconds + 1, int(sample_every)):
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
            tokens_all.extend(ocr_text(crop))

        id_code, cond = parse_labels(tokens_all)
        if id_code:
            id_counts[id_code] += 1
        if cond:
            cond_counts[cond] += 1

        # Save the first frame where we got both pieces (for auditing)
        if id_code and cond and save_frame_dir and not first_saved:
            new_base = f"{sanitize(id_code)}_{sanitize(cond)}"
            out_dir = Path(save_frame_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{new_base}_frame{frame_idx}.jpg"
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

# ---------- Main CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Extract labels from videos and build new filenames.")
    ap.add_argument("input_dir", type=Path, help="Folder containing videos")
    ap.add_argument("--seconds", type=int, default=100, help="How many seconds to scan from the start (default: 100)")
    ap.add_argument("--sample-every", type=float, default=1.0, help="Sample every N seconds (default: 1.0)")
    ap.add_argument("--rename", action="store_true", help="Rename videos in place when a name is found")
    ap.add_argument("--save-frames", type=Path, default=None, help="Folder to save the first good frame (optional)")
    ap.add_argument("--extensions", nargs="+",
                    default=[".mp4", ".avi", ".mov", ".mkv", ".m4v"],
                    help="Video file extensions to include")
    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input folder does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    mappings = []
    for p in sorted(args.input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in [e.lower() for e in args.extensions]:
            continue

        print(f"[INFO] Processing: {p.name}")
        new_base, saved = scan_video(p, seconds=args.seconds,
                                     sample_every=args.sample_every,
                                     save_frame_dir=args.save_frames)

        if new_base:
            new_name = f"{new_base}{p.suffix}"
            mappings.append((p.name, new_name))
            print(f"  -> Detected: {new_base}")
            if args.rename:
                target = p.with_name(new_name)
                # Avoid overwrite
                if target.exists():
                    # add numeric suffix
                    idx = 1
                    while True:
                        candidate = p.with_name(f"{new_base}_{idx}{p.suffix}")
                        if not candidate.exists():
                            target = candidate
                            break
                        idx += 1
                p.rename(target)
                print(f"  -> Renamed to: {target.name}")
        else:
            mappings.append((p.name, None))
            print("  -> No reliable labels detected")

        if saved:
            print(f"  -> Saved frame: {saved}")

    # Summary
    print("\n=== Summary ===")
    for orig, new in mappings:
        if new:
            print(f"{orig}  =>  {new}")
        else:
            print(f"{orig}  =>  (no match)")

if __name__ == "__main__":
    main()
