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
import sys
import subprocess


# ---------- helper: calculate folder size ----------
def get_folder_size(path):
    total_size = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
    return total_size


# ---------- run command safely ----------
def run(cmd, cwd=None):
    print(f"\n>> {cmd}")
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=cwd
    )
    return result.returncode


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Safe per-folder DataLad uploader"
    )
    parser.add_argument("-i", "--initial_path", required=True)
    parser.add_argument("-d", "--depth", required=True, type=int)
    args = parser.parse_args()

    initial_path = os.path.abspath(args.initial_path)
    depth = args.depth

    MIN_SIZE_BYTES = 5 * 1024 * 1024 * 1024

    print("------------------------------------------------------------")
    print(f"Dataset root: {initial_path}")
    print("------------------------------------------------------------")

    # ---------- build glob ----------
    search_path = initial_path
    for _ in range(depth):
        search_path = os.path.join(search_path, "*")

    folders = sorted(glob.glob(search_path))

    for path in folders:

        if not os.path.isdir(path):
            continue

        size = get_folder_size(path)
        if size < MIN_SIZE_BYTES:
            continue

        size_gb = size / (1024 ** 3)
        rel = os.path.relpath(path, initial_path)

        print(f"\n=== Processing: {rel} ({size_gb:.2f} GB) ===")

        # 1. save
        rc = run(f'datalad save "{path}" -m "Add folder: {rel}"')
        if rc != 0:
            print("ERROR: save failed. Aborting.")
            sys.exit(1)

        # 2. push (scoped)
        rc = run(f'datalad push --to origin "{path}"')
        if rc != 0:
            print("ERROR: push failed. Aborting to protect dataset.")
            sys.exit(1)

        # 3. drop (only after successful push)
        rc = run(f'datalad drop --what filecontent --recursive "{path}"')
        if rc != 0:
            print("ERROR: drop failed. Aborting.")
            sys.exit(1)

    print("\n---------------- FINISHED SAFELY ----------------")
