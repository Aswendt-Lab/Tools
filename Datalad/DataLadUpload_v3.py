"""
Created on Mon Nov 22 17:28:14 2021
@author: kalantaria
Modified: on Mon Feb 02 11:25:14 2026
@authro: maswendt
sequential datalad uploader
- operate only on folders >= 5 GB
- save per folder
- push once at the end
- drop only after push (safe git-annex workflow)
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


#%% Command line interface
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Sequential datalad uploader for folders >= 5GB'
    )
    parser.add_argument(
        '-i', '--initial_path',
        required=True,
        help='Path to the root dataset'
    )
    parser.add_argument(
        '-d', '--depth',
        required=True,
        type=int,
        help='Depth to search for subdirectories'
    )
    args = parser.parse_args()

    initial_path = os.path.abspath(args.initial_path)
    depth = args.depth

    MIN_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB

    print("Hello!")
    print('------------------------------------------------------------')
    print('Uploading folders >= 5 GB')
    print(f'Dataset root: {initial_path}')
    print('------------------------------------------------------------')

    # ---------- build glob pattern ----------
    search_path = initial_path
    for _ in range(depth):
        search_path = os.path.join(search_path, "*")

    candidates = glob.glob(search_path, recursive=True)
    print(f"Total paths found: {len(candidates)}")
    print('------------------------------------------------------------')

    saved_anything = False

    # ---------- save qualifying folders ----------
    for path in candidates:

        if not os.path.isdir(path):
            continue

        folder_size = get_folder_size(path)

        if folder_size < MIN_SIZE_BYTES:
            print(f"Skipping (too small): {path}")
            continue

        size_gb = folder_size / (1024 ** 3)
        rel_path = os.path.relpath(path, initial_path)

        print(f"\nSaving folder: {rel_path} ({size_gb:.2f} GB)")

        os.system(
            f'datalad save "{path}" -m "Add folder: {rel_path}"'
        )

        saved_anything = True

    # ---------- push once ----------
    if saved_anything:
        print('\nPushing dataset to origin...')
        os.system('datalad push --to origin')

        print('Dropping file content locally (safe)...')
        os.system('datalad drop --what filecontent --recursive .')
    else:
        print('\nNo folders met the size criterion. Nothing to push.')

    print('\n--------------------- FINISHED ------------------------------')
    print('------------------------------------------------------------')
