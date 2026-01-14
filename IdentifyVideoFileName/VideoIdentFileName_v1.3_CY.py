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
Yellow-card segmented, strict OCR → safe file names.
Default: COPY to <input>/renamed_copies (no overwrite). Use --rename to rename in place.

Examples
--------
python VideoIdentFileName.py /path/to/videos --seconds 60 --out-dir /path/to/output --behavior Cylinder
python VideoIdentFileName.py /path/to/videos --rename --behavior Rotarod
"""

import argparse, re, sys, shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pytesseract

# ---------------- OCR helpers ----------------

def rotate_bound(image, angle):
    (h, w) = image.shape[:2]; (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nW / 2) - cX; M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def preprocess_for_ocr(img_bgr):
    img = cv2.resize(img_bgr, None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.fastNlMeansDenoising(gray, h=8)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 9)
    return thr

def tess_tokens_multi(bin_img):
    """Try several Tesseract modes/whitelists; merge tokens."""
    tokens = []
    cfgs = [
        "--oem 3 --psm 6 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
        "--oem 3 --psm 7 -l eng -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_",
        "--oem 3 --psm 11 -l eng",
        "--oem 3 --psm 13 -l eng",
    ]
    for cfg in cfgs:
        data = pytesseract.image_to_data(bin_img, config=cfg, output_type=pytesseract.Output.DICT)
        for txt, conf in zip(data.get("text", []), data.get("conf", [])):
            if not txt:
                continue
            try: conf = float(conf)
            except: conf = -1.0
            if conf >= 55:
                tokens.append((txt.strip(), conf))
    return tokens

def ocr_multi_rotation(img_bgr):
    angles = [0, -20, -10, 10, 20, 180, 160, 170, 190, 200]
    scored = []
    for a in angles:
        r = rotate_bound(img_bgr, a) if a else img_bgr
        pre = preprocess_for_ocr(r)
        toks = tess_tokens_multi(pre)
        if toks:
            scored.append((sum(c for _, c in toks), toks))
    scored.sort(key=lambda x: x[0], reverse=True)
    merged = []
    for _, toks in scored[:2]:
        merged.extend(toks)
    return merged

# ---------------- Yellow-card detection ----------------

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1); rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def warp_from_box(img, box):
    rect = order_points(np.array(box, dtype="float32"))
    (tl, tr, br, bl) = rect
    wA = np.linalg.norm(br - bl); wB = np.linalg.norm(tr - tl)
    hA = np.linalg.norm(tr - br); hB = np.linalg.norm(tl - bl)
    maxW, maxH = int(max(wA, wB)), int(max(hA, hB))
    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    if warped.shape[0] > warped.shape[1]:
        warped = rotate_bound(warped, 90)
    return warped

def yellow_cards(frame_bgr):
    """Return warped crops of the yellow cards using HSV + LAB."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # wider yellow to catch “greenish” cards under cool light
    m1 = cv2.inRange(hsv, np.array([10, 40, 60]), np.array([45, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([45, 30, 60]), np.array([70, 255, 255]))  # yellow→yellow-green
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    b = lab[:, :, 2]
    _, m3 = cv2.threshold(b, 150, 255, cv2.THRESH_BINARY)  # yellow has high b-channel
    mask = cv2.bitwise_or(cv2.bitwise_or(m1, m2), m3)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = frame_bgr.shape[:2]
    min_area = max(1200, int(0.0003 * H * W))
    crops = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        crops.append(warp_from_box(frame_bgr, box))
    return crops

# ---------------- Parsing (strict but tolerant) ----------------

# ID: e.g., GV_T3_10_4  (>= 2 underscores, total length >= 9)
ID_STRICT = re.compile(r"^[A-Z]{1,4}_[A-Z]\d+(?:_\d+){2,6}$")
PIECE = re.compile(r"^[A-Za-z0-9]{1,4}$")

COND_PATTS = [
    re.compile(r"^Baseline$", re.I),
    re.compile(r"^Pre$", re.I),
    re.compile(r"^Post$", re.I),
    re.compile(r"^Control$", re.I),
    re.compile(r"^P\s*[_-]?\s*\d{1,3}$", re.I),
    re.compile(r"^D\s*[_-]?\s*\d{1,3}$", re.I),
    re.compile(r"^Day\s*[_-]?\s*\d{1,3}$", re.I),
    re.compile(r"^W\s*[_-]?\s*\d{1,2}$", re.I),
]

def reconstruct_id_from_pieces(tokens):
    pieces = [t[0].replace("-", "_").replace("__", "_") for t in tokens]
    n = len(pieces); cands = []
    for i in range(n):
        if not PIECE.fullmatch(pieces[i]): continue
        if not re.match(r"^[A-Z]{1,4}[A-Z0-9]*$", pieces[i], re.I): continue
        cur = [pieces[i].upper()]
        for j in range(i+1, min(i+7, n)):
            if PIECE.fullmatch(pieces[j]):
                cur.append(pieces[j].upper())
                cand = "_".join(cur)
                if cand.count("_") >= 2 and len(cand) >= 9 and ID_STRICT.match(cand):
                    cands.append(cand)
            else:
                break
    return cands

def parse_card(tokens):
    """Return (id_candidate or None, cond_candidate or None) for a single card."""
    raw = " ".join(t[0] for t in tokens).upper()
    flat = re.sub(r"[^A-Z0-9_]", "_", raw)

    # strict ID directly
    for s in set(re.split(r"\s+", flat)):
        s = s.strip("_")
        if ID_STRICT.match(s):
            return s, None

    # reconstruct within card
    cands = reconstruct_id_from_pieces(tokens)
    if cands:
        cands.sort(key=lambda s: (s.count("_"), len(s)), reverse=True)
        return cands[0], None

    # condition tags
    for patt in COND_PATTS:
        m = patt.search(raw.replace(" ", ""))
        if m:
            cond = m.group(0).replace(" ", "").replace("-", "").replace("_", "")
            cond = cond.capitalize() if cond.lower() in {"baseline","pre","post","control"} else cond
            return None, cond

    # loose fallback for P## on the card
    m = re.search(r"\bP\d{1,3}\b", raw, re.I)
    if m: return None, m.group(0).upper()
    return None, None

def sanitize(s): return re.sub(r"[^A-Za-z0-9_]+", "", s.replace(" ", "_"))

# ---------------- Video scanning ----------------

def scan_video(path: Path, seconds=100, sample_every=1.0, save_frame_dir: Path|None=None):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open: {path}", file=sys.stderr); return None, None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = total / fps if fps > 0 else seconds
    max_seconds = min(seconds, int(duration))

    id_counts, cond_counts = Counter(), Counter()
    first_saved = None
    step = max(1, int(sample_every))

    for sec in range(0, max_seconds + 1, step):
        if fps <= 0: break
        frame_idx = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None: continue

        got_any = False
        for card in yellow_cards(frame):
            toks = ocr_multi_rotation(card)
            i, c = parse_card(toks)
            if i: id_counts[i] += 1; got_any = True
            if c: cond_counts[c] += 1; got_any = True

        # Fallback only if no cards found at all
        if not got_any:
            h, w = frame.shape[:2]
            for x0f, x1f in [(0, .33), (.66, 1), (0, 1)]:
                x0, x1 = int(w*x0f), int(w*x1f)
                toks = ocr_multi_rotation(frame[:, x0:x1])
                i, c = parse_card(toks)
                if i: id_counts[i] += 1
                if c: cond_counts[c] += 1

        if id_counts and cond_counts and save_frame_dir and not first_saved:
            id_best = id_counts.most_common(1)[0][0]
            cond_best = cond_counts.most_common(1)[0][0]
            save_frame_dir.mkdir(parents=True, exist_ok=True)
            out = save_frame_dir / f"{sanitize(id_best)}_{sanitize(cond_best)}_frame{frame_idx}.jpg"
            cv2.imwrite(str(out), frame); first_saved = out

    cap.release()

    id_best = id_counts.most_common(1)[0][0] if id_counts else None
    cond_best = cond_counts.most_common(1)[0][0] if cond_counts else None
    if id_best and cond_best: return f"{sanitize(id_best)}_{sanitize(cond_best)}", first_saved
    if id_best:  return sanitize(id_best), first_saved
    if cond_best: return sanitize(cond_best), first_saved
    return None, first_saved

# ---------------- File ops & CLI ----------------

def unique_target(out_dir: Path, base: str, ext: str) -> Path:
    p = out_dir / f"{base}{ext}"
    if not p.exists(): return p
    i = 1
    while True:
        cand = out_dir / f"{base}_{i}{ext}"
        if not cand.exists(): return cand
        i += 1

def main():
    ap = argparse.ArgumentParser(description="Extract yellow-card labels and safely rename/copy videos.")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("--seconds", type=int, default=100)
    ap.add_argument("--sample-every", dest="sample_every", type=float, default=1.0)
    ap.add_argument("--save-frames", dest="save_frames", type=Path, default=None)
    ap.add_argument("--extensions", nargs="+", default=[".mp4", ".avi", ".mov", ".mkv", ".m4v"])

    # New: add behavior/test tag at the end, e.g. _Cylinder
    ap.add_argument("--behavior", type=str, default=None,
                    help="Behavior/test name to append to the new file name, e.g. 'Cylinder'")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--rename", action="store_true", help="Rename in place (never overwrite).")
    g.add_argument("--copy", action="store_true", help="Copy to --out-dir (default).")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output folder for copies (default: <input>/renamed_copies)")

    args = ap.parse_args()
    if not args.input_dir.exists():
        print(f"[ERROR] Input folder does not exist: {args.input_dir}", file=sys.stderr); sys.exit(1)

    do_copy = args.copy or not args.rename
    out_dir = args.out_dir or (args.input_dir / "renamed_copies")
    if do_copy: out_dir.mkdir(parents=True, exist_ok=True)

    mappings = []
    for p in sorted(args.input_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in [e.lower() for e in args.extensions]:
            continue

        print(f"[INFO] Processing: {p.name}")
        base, saved = scan_video(p, seconds=args.seconds, sample_every=args.sample_every, save_frame_dir=args.save_frames)

        if base:
            # append behavior tag if provided
            if args.behavior:
                safe_beh = re.sub(r"[^A-Za-z0-9]+", "", args.behavior)
                if safe_beh:
                    base = f"{base}_{safe_beh}"

            if do_copy:
                target = unique_target(out_dir, base, p.suffix)
                shutil.copy2(p, target)
                print(f"  -> Copied to: {target.name}")
            else:
                target = unique_target(p.parent, base, p.suffix)
                p.rename(target)
                print(f"  -> Renamed to: {target.name}")
            mappings.append((p.name, target.name))
        else:
            print("  -> No reliable labels detected.")
            mappings.append((p.name, None))

        if saved:
            print(f"  -> Saved frame: {saved}")

    print("\n=== Summary ===")
    for src, dst in mappings:
        print(f"{src}  =>  {dst if dst else '(no match)'}")

if __name__ == "__main__":
    main()
