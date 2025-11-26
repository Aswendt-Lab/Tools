""""
Created on 26.11.2025

@authors: Markus Aswendt, ChatGPT
Department of Neurology
University Hospital Frankfurt
Theodor-Stern-Kai 7
D-60590 Frankfurt am Main

"""

#!/usr/bin/env python3
"""
Orientation- and color-aware label extractor.
Same CLI/behavior as before: copies by default (no overwrite), or --rename.

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

# ---------- OCR & image helpers ----------

def rotate_bound(image, angle_degrees: float):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle_degrees, 1.0)
    cos = abs(M[0, 0]); sin = abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_for_ocr(img_bgr):
    img = cv2.resize(img_bgr, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=8)
    thr = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)
    return thr

def _tess_tokens(bin_img):
    cfg = "--oem 3 --psm 6 -l eng"
    data = pytesseract.image_to_data(bin_img, config=cfg, output_type=pytesseract.Output.DICT)
    toks = []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        if not txt:
            continue
        try: conf = float(conf)
        except: conf = -1.0
        if conf >= 55:
            toks.append((txt.strip(), conf))
    return toks

def ocr_text_multi_rotation(img_bgr):
    angles = [0, -20, -10, 10, 20, 180, 160, 170, 190, 200]
    best = []
    for ang in angles:
        r = rotate_bound(img_bgr, ang) if ang else img_bgr
        pre = preprocess_for_ocr(r)
        toks = _tess_tokens(pre)
        if toks:
            score = sum(c for _, c in toks)
            best.append((score, toks))
    best.sort(key=lambda x: x[0], reverse=True)
    merged = []
    for _, toks in best[:2]:
        merged.extend(toks)
    return merged

def order_points(pts):
    # pts: 4x2
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]                      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]                   # top-right
    rect[3] = pts[np.argmax(diff)]                   # bottom-left
    return rect

def warp_from_box(img, box):
    # box: 4x2 points from cv2.boxPoints(minAreaRect)
    rect = order_points(np.array(box, dtype="float32"))
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl); widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br); heightB = np.linalg.norm(tl - bl)
    maxW = int(max(widthA, widthB))
    maxH = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # Ensure landscape orientation for OCR
    if warped.shape[0] > warped.shape[1]:
        warped = rotate_bound(warped, 90)
    return warped

def yellow_card_tokens(frame_bgr):
    """
    Find yellow rectangles, warp them flat, and OCR them with multi-rotation.
    Returns a list of tokens aggregated across all detected cards.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    # fairly generous yellow range
    lower1 = np.array([15, 60, 80]); upper1 = np.array([40, 255, 255])
    mask = cv2.inRange(hsv, lower1, upper1)

    # clean up
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = frame_bgr.shape[:2]
    min_area = max(1500, int(0.0005 * H * W))
    tokens = []

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        warped = warp_from_box(frame_bgr, box)
        tokens.extend(ocr_text_multi_rotation(warped))

    return tokens

# ---------- Parsing ----------

ID_REGEX   = re.compile(r"\b[A-Z]{1,4}(?:_[A-Z0-9]+){1,6}\b")
COND_PATTS = [
    re.compile(r"\bBaseline\b", re.I),
    re.compile(r"\bPre\b", re.I),
    re.compile(r"\bPost\b", re.I),
    re.compile(r"\bControl\b", re.I),
    re.compile(r"\bP\s*[_-]?\s*\d{1,3}\b", re.I),   # P20, P_22, P-5
    re.compile(r"\bD\s*[_-]?\s*\d{1,3}\b", re.I),   # D1, D_14
    re.compile(r"\bDay\s*[_-]?\s*\d{1,3}\b", re.I), # Day1, Day_7
    re.compile(r"\bW\s*[_-]?\s*\d{1,2}\b", re.I),   # W2 (week)
]

def parse_labels(tokens):
    texts = [t[0] for t in tokens]
    joined = " ".join(texts)

    # ID-like code (with underscores)
    ids = ID_REGEX.findall(joined)
    id_code = Counter(ids).most_common(1)[0][0] if ids else None

    # Condition: try patterns in priority order
    cond = None
    for patt in COND_PATTS:
        m = patt.search(joined)
        if m:
            cond = m.group(0).replace(" ", "").replace("-", "").replace("__", "_")
            cond = cond.replace("_", "")  # keep simple like P20, Day1
            cond = cond.capitalize() if cond.lower() in {"baseline","pre","post","control"} else cond
            break

    # Fallback: any short alnum token that looks like a tag (e.g., P20)
    if not cond:
        alnums = [t for t in texts if re.fullmatch(r"[A-Za-z]\d{1,3}", t)]
        if alnums:
            cond = Counter(alnums).most_common(1)[0][0]

    return id_code, cond

def sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "", s.replace(" ", "_"))

# ---------- Video processing ----------

def scan_video(path: Path, seconds=100, sample_every=1.0, save_frame_dir: Path | None=None):
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

    step = max(1, int(sample_every))
    for sec in range(0, max_seconds + 1, step):
        if fps <= 0:
            break
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        # 1) Prefer OCR from detected yellow cards
        tokens_all = yellow_card_tokens(frame)

        # 2) If nothing found, fall back to coarse crops & rotations
        if not tokens_all:
            h, w = frame.shape[:2]
            for x0f, x1f in [(0.00,0.33),(0.66,1.00),(0.0,1.0)]:
                x0 = int(w * x0f); x1 = int(w * x1f)
                crop = frame[:, x0:x1]
                tokens_all.extend(ocr_text_multi_rotation(crop))

        id_code, cond = parse_labels(tokens_all)
        if id_code: id_counts[id_code] += 1
        if cond:    cond_counts[cond] += 1

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
    target = out_dir / f"{base}{ext}"
    if not target.exists():
        return target
    i = 1
    while True:
        candidate = out_dir / f"{base}_{i}{ext}"
        if not candidate.exists():
            return candidate
        i += 1

# ---------- Main CLI (unchanged) ----------

def main():
    ap = argparse.ArgumentParser(description="Extract labels (yellow-card robust) and create safe new names.")
    ap.add_argument("input_dir", type=Path, help="Folder containing videos")
    ap.add_argument("--seconds", type=int, default=100, help="How many seconds to scan from the start (default: 100)")
    ap.add_argument("--sample-every", dest="sample_every", type=float, default=1.0,
                    help="Sample every N seconds (default: 1.0)")
    ap.add_argument("--save-frames", dest="save_frames", type=Path, default=None,
                    help="Folder to save the first good frame (optional)")
    ap.add_argument("--extensions", nargs="+",
                    default=[".mp4", ".avi", ".mov", ".mkv", ".m4v"],
                    help="Video file extensions to include")

    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--rename", action="store_true",
                      help="Rename originals in place (non-overwriting).")
    mode.add_argument("--copy", action="store_true",
                      help="Copy to --out-dir (default behavior).")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output folder for copies (default: <input>/renamed_copies)")

    args = ap.parse_args()

    if not args.input_dir.exists():
        print(f"[ERROR] Input folder does not exist: {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    do_copy = args.copy or not args.rename
    out_dir = args.out_dir or (args.input_dir / "renamed_copies")
    if do_copy:
        out_dir.mkdir(parents=True, exist_ok=True)

    mappings = []
    for p in sorted(args.input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in [e.lower() for e in args.extensions]:
            continue

        print(f"[INFO] Processing: {p.name}")
        new_base, saved = scan_video(
            p, seconds=args.seconds, sample_every=args.sample_every, save_frame_dir=args.save_frames
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

    print("\n=== Summary ===")
    for orig, new in mappings:
        print(f"{orig}  =>  {new if new else '(no match)'}")

if __name__ == "__main__":
    main()
