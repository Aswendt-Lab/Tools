import sys
import shutil
import re
import json
from pathlib import Path
from typing import Any, Optional

# ------------------------------------------------------------
# Helper files to move to sourcedata/
# ------------------------------------------------------------

HELPER_FILES = {
    "conv2nifti_log.txt",
    "dataset.csv",
    "GroupMapping.xlsx",
    "dataset.json",
    "README",
}

# ------------------------------------------------------------
# 1. Move helper files
# ------------------------------------------------------------

def move_helper_files(root: Path) -> None:
    sd = root / "sourcedata"
    sd.mkdir(exist_ok=True)
    for name in HELPER_FILES:
        src = root / name
        if src.exists():
            dst = sd / src.name
            print(f"[helper] Moving {src} → {dst}")
            shutil.move(str(src), str(dst))


# ------------------------------------------------------------
# 2. Fix stray top-level ses-* directories
# ------------------------------------------------------------

def infer_subject_from_files(ses_dir: Path) -> Optional[str]:
    for f in ses_dir.rglob("*"):
        if not f.is_file():
            continue
        m = re.search(r"sub-([^_]+)_ses-", f.name)
        if m:
            return m.group(1)
    return None


def fix_top_level_sessions(root: Path):
    for ses_dir in root.glob("ses-*"):
        if not ses_dir.is_dir():
            continue

        subj = infer_subject_from_files(ses_dir)
        if subj is None:
            continue

        subj_dir = root / f"sub-{subj}"
        if subj_dir.exists():
            dst = subj_dir / ses_dir.name
            if not dst.exists():
                print(f"[ses] Moving {ses_dir} → {dst}")
                shutil.move(str(ses_dir), str(dst))


# ------------------------------------------------------------
# 3. Rename fMRI EPI → *_task-rest_bold
# ------------------------------------------------------------

def rename_epi_to_bold(root: Path, task="rest"):
    for func_dir in root.glob("sub-*/ses-*/func"):
        for nii in func_dir.glob("*_EPI.nii.gz"):
            base = nii.name
            new_base = base.replace("_EPI", f"_task-{task}_bold")
            new_nii = nii.with_name(new_base)

            json_old = nii.with_suffix("").with_suffix(".json")
            json_new = new_nii.with_suffix("").with_suffix(".json")

            print(f"[func] {nii.name} → {new_nii.name}")
            nii.rename(new_nii)

            if json_old.exists():
                json_old.rename(json_new)


def fix_repetition_time(meta: dict) -> bool:
    """
    Convert RepetitionTime from ms to seconds if it looks like it's in ms.
    Heuristic: if RepetitionTime > 50, assume it's ms and divide by 1000.
    Returns True if changed.
    """
    key = "RepetitionTime"
    val = meta.get(key)
    if isinstance(val, (int, float)) and val > 50:
        meta[key] = val / 1000.0
        return True
    return False

# ------------------------------------------------------------
# 4. DELETE broken fieldmaps (< 1 KB)
# ------------------------------------------------------------

def remove_corrupted_fieldmaps(root: Path):
    for nii in root.rglob("*_fieldmap.nii.gz"):
        if nii.stat().st_size < 1024:
            print(f"[fmap] Removing CORRUPTED fieldmap {nii}")
            json_sidecar = nii.with_suffix("").with_suffix(".json")
            nii.unlink()
            if json_sidecar.exists():
                json_sidecar.unlink()

    for nii in root.rglob("*_magnitude.nii.gz"):
        if nii.stat().st_size < 1024:
            print(f"[fmap] Removing CORRUPTED magnitude {nii}")
            json_sidecar = nii.with_suffix("").with_suffix(".json")
            nii.unlink()
            if json_sidecar.exists():
                json_sidecar.unlink()


# ------------------------------------------------------------
# 5. JSON cleaning functions
# ------------------------------------------------------------

def clean_json(json_path: Path):
    try:
        meta = json.loads(json_path.read_text())
    except:
        return

    changed = False

    # set TaskName for bold files
    if json_path.name.endswith("_bold.json"):
        if "TaskName" not in meta:
            meta["TaskName"] = "rest"
            changed = True

    # TR fix (ms → sec)
    if fix_repetition_time(meta):
        changed = True

    # Generic cleaning
    if "DeviceSerialNumber" in meta and not isinstance(meta["DeviceSerialNumber"], str):
        meta["DeviceSerialNumber"] = str(meta["DeviceSerialNumber"])
        changed = True

    if "NonlinearGradientCorrection" in meta and not isinstance(meta["NonlinearGradientCorrection"], bool):
        del meta["NonlinearGradientCorrection"]
        changed = True

    if "PartialFourier" in meta and not isinstance(meta["PartialFourier"], (int, float)):
        del meta["PartialFourier"]
        changed = True

    if "InversionTime" in meta and not isinstance(meta["InversionTime"], (int, float)):
        del meta["InversionTime"]
        changed = True

    if "ScanOptions" in meta and not isinstance(meta["ScanOptions"], (str, list)):
        del meta["ScanOptions"]
        changed = True

    if changed:
        json_path.write_text(json.dumps(meta, indent=4))
        print(f"[json] Updated {json_path}")


# ------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------

def main():
    if len(sys.argv) != 2:
        print("Usage: python mouse_bids_cleanup_final.py /path/to/mouse_cleanup")
        return

    root = Path(sys.argv[1]).resolve()
    print(f"[*] BIDS root: {root}")

    move_helper_files(root)
    fix_top_level_sessions(root)
    rename_epi_to_bold(root)
    remove_corrupted_fieldmaps(root)

    for json_path in root.rglob("*.json"):
        clean_json(json_path)

    print("[*] Final cleanup done — re-run bids-validator.")

if __name__ == "__main__":
    main()
