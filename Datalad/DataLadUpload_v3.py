"""
Created on Mon Nov 22 17:28 2021
@author: kalantaria
Modified: on Mon Feb 03 13:08 2026
@authors: maswendt, ChatGPT
sequential datalad uploader
- operates on folders >= 5 GB
- per-folder: save -> push -> drop
- safe git-annex behavior
"""

import argparse
import os
import glob

# ---------- helper: calculate folder size ----------
def get_folder_size(path):
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size  # bytes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Per-folder DataLad uploader (>= 5GB)"
    )
    parser.add_argument(
        "-i", "--initial_path",
        required=True,
        help="Path to dataset root"
    )
    parser.add_argument(
        "-d", "--depth",
        required=True,
        type=int,
        help="Folder depth to search"
    )
    args = parser.parse_args()

    initial_path = os.path.abspath(args.initial_path)
    depth = args.depth

    MIN_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

    print("------------------------------------------------------------")
    print(f"Dataset root: {initial_path}")
    print("------------------------------------------------------------")

    # ---------- build glob pattern ----------
    search_path = initial_path
    for _ in range(depth):
        search_path = os.path.join(search_path, "*")

    candidates = sorted(glob.glob(search_path))

    for path in candidates:

        if not os.path.isdir(path):
            continue

        folder_size = get_folder_size(path)
        if folder_size < MIN_SIZE_BYTES:
            continue

        size_gb = folder_size / (1024 ** 3)
        rel_path = os.path.relpath(path, initial_path)

        print(f"\nProcessing folder: {rel_path} ({size_gb:.2f} GB)")

        # 1️⃣ save
        os.system(
            f'datalad save "{path}" -m "Add folder: {rel_path}"'
        )

        # 2️⃣ push ONLY this folder
        os.system(
            f'datalad push --to origin "{path}"'
        )

        # 3️⃣ drop ONLY this folder (safe)
        os.system(
            f'datalad drop --what filecontent --recursive "{path}"'
        )

    print("\n--------------------- FINISHED ------------------------------")
