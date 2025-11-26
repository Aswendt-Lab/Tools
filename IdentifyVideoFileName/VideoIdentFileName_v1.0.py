""""
Created on 05.11.2025

@authors: Markus Aswendt, ChatGPT
Department of Neurology
University Hospital Frankfurt
Theodor-Stern-Kai 7
D-60590 Frankfurt am Main

"""

#!/usr/bin/env python3
# VideoIdentFileName.py
import argparse, re
from pathlib import Path
import cv2
import numpy as np
import pytesseract

# If you use Homebrew's tesseract instead of conda's, uncomment the right path:
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"   # Apple Silicon
# pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"      # Intel macOS

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv"}

def is_video(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS

# ---------------- OCR helpers ----------------

def clean(s: str) -> str:
    s = s.strip().replace("\n", " ")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "", s)
    return s.upper()

def fix_confusions(text: str, kind: str) -> str:
    """Lightweight character repair for GV and STAGE labels."""
    t = text.upper()
    # shared
    t = t.replace("—", "_").replace("-", "_")
    t = re.sub(r"_+", "_", t)

    if kind == "GV":
        # common digit/letter confusions when GV label has underscores + numbers
        t = t.replace("O", "0").replace("I", "1").replace("L", "1").replace("|", "1")
        t = t.replace("S", "5").replace("B", "8").replace("Z", "2")
        # enforce leading GV
        if "6V" in t: t = t.replace("6V", "GV")
        if not t.startswith("GV") and "GV" in t:
            i = t.index("GV")
            t = t[i:]
        # collapse accidental double underscores
        t = re.sub(r"_{2,}", "_", t)
        # keep only GV, T, digits and underscores
        t = re.sub(r"[^GVT0-9_]", "", t)
    else:  # STAGE
        # normalize baseline / P#
        t = t.replace("BA5ELINE", "BASELINE").replace("8ASELINE", "BASELINE")
        t = t.replace("BASELlNE", "BASELINE").replace("BAS ELINE", "BASELINE")
        t = t.replace("O", "0")  # P0 vs P O
        t = re.sub(r"[^A-Z0-9_]", "", t)
        # prefer a clear pattern if present
        m = re.search(r"P\d{1,2}[A-Z]?", t)
        if m: t = m.group(0)
        elif "BASELINE" in t:
            t = "BASELINE"
    return t

def ocr_single_line(img: np.ndarray, whitelist: str) -> str:
    config = f'--psm 7 -c tessedit_char_whitelist={whitelist}'
    return pytesseract.image_to_string(img, config=config)

def ocr_label(crop_bgr: np.ndarray, kind: str) -> str:
    """Deskew + threshold + OCR for a single label area."""
    # grayscale + gentle denoise
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 7, 75, 75)
    # contrast/stretch & threshold
    g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
    thr = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 41, 15)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), 1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), 1)
    thr = cv2.resize(thr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    if kind == "GV":
        wl = "GVT0123456789_"
    else:
        wl = "P0123456789BASELINE_"

    txt = clean(ocr_single_line(thr, wl))
    if not txt:
        # try a different page segmentation as fallback
        txt = clean(pytesseract.image_to_string(thr, config=f'--psm 6 -c tessedit_char_whitelist={wl}'))
    return fix_confusions(txt, kind)

# -------------- detection helpers --------------

def crop_rotated_rect(img: np.ndarray, rect):
    """Crop a rotated rectangle (minAreaRect) into an upright ROI."""
    (cx, cy), (w, h), angle = rect
    if w < 1 or h < 1:
        return None
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, rot_mat, (img.shape[1], img.shape[0]))
    w, h = int(w), int(h)
    x0, y0 = int(cx - w/2), int(cy - h/2)
    x1, y1 = x0 + w, y0 + h
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(rotated.shape[1], x1); y1 = min(rotated.shape[0], y1)
    return rotated[y0:y1, x0:x1]

def find_labels(frame_bgr: np.ndarray, roi_left_frac=0.6):
    """
    Restrict search to the left part of the image (labels live there in your setup).
    Return up to two deskewed crops + rectangles sorted top->bottom.
    """
    H, W, _ = frame_bgr.shape
    Wcut = int(W * roi_left_frac)
    roi = frame_bgr[:, :Wcut]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # tighter yellow to reject white glare
    lower = np.array([18, 80, 70], dtype=np.uint8)
    upper = np.array([42, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in cnts:
        rect = cv2.minAreaRect(c)  # ((cx,cy),(w,h),angle)
        (cx, cy), (w, h), ang = rect
        area = w * h
        if area < 2500:  # ignore tiny bits
            continue
        ar = max(w, h) / (min(w, h) + 1e-6)
        if ar < 0.7 or ar > 4.2:   # sticky-note-ish
            continue

        # crop deskewed
        crop = crop_rotated_rect(roi, rect)
        if crop is None or crop.size == 0:
            continue

        # pad a little to include full text
        pad = 8
        y0 = max(0, int(cy - h/2) - pad)
        y1 = min(roi.shape[0], int(cy + h/2) + pad)
        x0 = max(0, int(cx - w/2) - pad)
        x1 = min(roi.shape[1], int(cx + w/2) + pad)
        crop = roi[y0:y1, x0:x1]

        # keep with global coords for debugging
        candidates.append(((int(cx), int(cy)), crop, ((x0, y0), (x1, y1))))

    # sort by Y position (top to bottom)
    candidates.sort(key=lambda k: k[0][1])
    return candidates[:2], roi

def pick_roles(txt1: str, txt2: str):
    gv_re = re.compile(r"^GV[_A-Z0-9]+$")
    p_re  = re.compile(r"^P\d{1,2}[A-Z]?$")
    b_re  = re.compile(r"^BASELINE$")

    score = lambda t: (bool(gv_re.match(t)), bool(p_re.match(t) or b_re.match(t)))

    a_gv, a_stage = score(txt1)
    b_gv, b_stage = score(txt2)

    if a_gv and b_stage: return txt1, txt2
    if b_gv and a_stage: return txt2, txt1
    # fallback: if one contains GV, make that first
    if "GV" in txt1 and "GV" not in txt2: return txt1, txt2
    if "GV" in txt2 and "GV" not in txt1: return txt2, txt1
    # final fallback: keep order
    return txt1, txt2

# -------------- main pipeline --------------

def extract_from_video(video: Path, sample_every_s=2.0, max_samples=1200, roi_left_frac=0.6, debug_dir: Path|None=None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames == 0:
        total_frames = int(10*60*fps)  # safety fallback

    step = max(int(sample_every_s * fps), 1)
    frames = list(range(0, total_frames, step))[:max_samples]

    last_debug = None
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok or frame is None: 
            continue

        labels, roi = find_labels(frame, roi_left_frac=roi_left_frac)
        if len(labels) < 2:
            continue

        # OCR both crops
        txt_top  = ocr_label(labels[0][1], kind="GV")     # try GV assumptions first
        txt_bot  = ocr_label(labels[1][1], kind="STAGE")

        # If either looks empty, retry with opposite assumptions
        if not txt_top or txt_top == "GV":
            txt_top = ocr_label(labels[0][1], kind="STAGE")
        if not txt_bot or txt_bot == "P":
            txt_bot = ocr_label(labels[1][1], kind="GV")

        a, b = pick_roles(txt_top, txt_bot)

        # Make a debug overlay
        debug = frame.copy()
        for i, ((cx, cy), _, ((x0, y0), (x1, y1))) in enumerate(labels):
            color = (0,255,0) if i == 0 else (255,0,0)
            cv2.rectangle(debug, (x0, y0), (x1, y1), color, 2)
        cv2.putText(debug, f"A:{txt_top}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.putText(debug, f"B:{txt_bot}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
        last_debug = debug

        # Accept if both are plausible and different
        if a and b and a != b:
            cap.release()
            if debug_dir:
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str((debug_dir / f"{video.stem}_debug.png")), last_debug)
                cv2.imwrite(str((debug_dir / f"{video.stem}_A_crop.png")), labels[0][1])
                cv2.imwrite(str((debug_dir / f"{video.stem}_B_crop.png")), labels[1][1])
            return a, b

    cap.release()
    if debug_dir and last_debug is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str((debug_dir / f"{video.stem}_debug_last.png")), last_debug)
    return None, None

def rename_video(video: Path, sample_every, max_samples, roi_left_frac, dry, overwrite, debug_dir):
    top, bottom = extract_from_video(video, sample_every, max_samples, roi_left_frac, debug_dir)
    if not top or not bottom:
        print(f"[SKIP] Could not read labels for: {video}")
        return False
    new_name = f"{top}_{bottom}{video.suffix.lower()}"
    target = video.with_name(new_name)

    if target.exists() and not overwrite:
        if target.resolve() == video.resolve():
            print(f"[OK] Already named: {video.name}")
            return True
        print(f"[SKIP] Target exists (use --overwrite): {target.name}")
        return False

    if dry:
        print(f"[DRY] {video.name} -> {new_name}")
        return True

    try:
        video.rename(target)
        print(f"[RENAMED] {video.name} -> {new_name}")
        return True
    except Exception as e:
        print(f"[ERROR] {video}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser(description="Rename videos by reading yellow labels (GV_* and P#/BASELINE).")
    ap.add_argument("root", type=Path, help="Folder to scan recursively")
    ap.add_argument("--sample-every", type=float, default=2.0, help="Seconds between sampled frames")
    ap.add_argument("--max-samples", type=int, default=1200, help="Max frames to probe per video")
    ap.add_argument("--roi-left-frac", type=float, default=0.6, help="Search only left fraction of frame (0–1)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", type=Path, default=None, help="Write debug PNGs to this folder")
    args = ap.parse_args()

    files = [p for p in args.root.rglob("*") if is_video(p)]
    print(f"Found {len(files)} video(s). Processing...")
    for v in files:
        rename_video(v, args.sample_every, args.max_samples, args.roi_left_frac, args.dry_run, args.overwrite, args.debug)

if __name__ == "__main__":
    main()
