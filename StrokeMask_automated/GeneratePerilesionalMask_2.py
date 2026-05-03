"""
@author: Motahare

This script:
1. Recursively renames existing raw stroke mask files according to a session mapping.
2. Creates perilesional stroke masks for acute sessions (ses-P1 to ses-P5) by dilating and subtracting the raw stroke mask.
3. Creates perilesional stroke masks for chronic sessions (e.g., ses-P28) by subtracting the local chronic stroke mask from the acute perilesional mask.
4. Registers each perilesional mask across all timepoints for each subject.
5. Applies whole-brain mask correction only to the perilesional stroke masks, ensuring no voxels outside the brain.

"""

import os
import glob
import argparse
import re
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

# ----------------------------------------------------------------------------
# SESSION MAPPING
# Maps all acute sessions (e.g., ses-P1, ses-P2) to ses-P3,
# and all chronic sessions (e.g., ses-P29, ses-P30) to ses-P28, etc.
# ----------------------------------------------------------------------------
session_mapping = {
    "ses-P1": "ses-P3",
    "ses-P2": "ses-P3",
    "ses-P3": "ses-P3",
    "ses-P4": "ses-P3",
    "ses-P5": "ses-P3",
    "ses-P27": "ses-P28",
    "ses-P28": "ses-P28",
    "ses-P29": "ses-P28",
    "ses-P30": "ses-P28",
    "ses-P42": "ses-P42",
    "ses-P43": "ses-P42",
    "ses-P56": "ses-P56",
    "ses-P57": "ses-P56",
    "ses-P58": "ses-P56",
}

# ----------------------------------------------------------------------------
# FUNCTION: rename_stroke_masks
# Renames raw stroke mask files (<subject>_<ses>-Stroke_mask.nii.gz)
# according to the session_mapping dictionary.
# ----------------------------------------------------------------------------
def rename_stroke_masks(input_path):
    anat_dirs = glob.glob(os.path.join(input_path, "*", "*", "anat"))
    for anat_dir in anat_dirs:
        # Iterate through all files in anat folder
        for filename in os.listdir(anat_dir):
            # Match pattern: prefix_ses-PX_Stroke_mask.nii.gz
            match = re.match(r"(.+?)_(ses-P\d+)_Stroke_mask\.nii\.gz", filename)
            if not match:
                continue

            prefix, sess = match.groups()
            if sess not in session_mapping:
                # Skip sessions we don't map
                continue

            mapped = session_mapping[sess]
            if sess == mapped or mapped in filename:
                # Already renamed or mapping to itself
                continue

            old_path = os.path.join(anat_dir, filename)
            new_filename = filename.replace(sess, mapped)
            new_path = os.path.join(anat_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Renamed {filename} -> {new_filename}")

# ----------------------------------------------------------------------------
# FUNCTION: create_perilesional_mask_no_stroke
# Dilates the raw stroke mask by `iterations` and subtracts the original
# to yield a perilesional ring mask (acute sessions).
# ----------------------------------------------------------------------------
def create_perilesional_mask_no_stroke(mask_path, iterations=1):
    img = nib.load(mask_path)
    data = img.get_fdata()
    original_mask = data > 0  # binary stroke mask

    # 3×3×3 structuring element for dilation
    structure = np.ones((3, 3, 3))
    dilated_mask = original_mask.copy()
    for _ in range(iterations):
        dilated_mask = binary_dilation(dilated_mask, structure=structure)

    # Perilesional region = dilated minus original
    perilesional = np.logical_and(dilated_mask, ~original_mask).astype(np.uint8)
    return nib.Nifti1Image(perilesional, img.affine, img.header)

# ----------------------------------------------------------------------------
# FUNCTION: create_perilesional_mask_sesP28
# For chronic sessions (mapped to ses-P28), subtracts the local chronic
# stroke mask (*BiasBetStroke_mask) from the pre-registered acute perilesional.
# ----------------------------------------------------------------------------
def create_perilesional_mask_sesP28(pre_mask_path, anat_dir, subject, timepoint, output_dir):
    # Load the pre-registered perilesional mask (acute-derived)
    pre_img = nib.load(pre_mask_path)
    pre_data = pre_img.get_fdata() > 0

    # Find the local raw stroke mask for this chronic session
    stroke_files = glob.glob(os.path.join(anat_dir, "*BiasBetStroke_mask.nii.gz"))
    if not stroke_files:
        print(f"No local stroke mask found in {anat_dir} for {timepoint}.")
        return None, None

    stroke_img = nib.load(stroke_files[0])
    stroke_data = stroke_img.get_fdata() > 0

    # Subtract local chronic stroke from the acute perilesional
    corrected = np.logical_and(pre_data, ~stroke_data).astype(np.uint8)
    new_img = nib.Nifti1Image(corrected, pre_img.affine, pre_img.header)

    # Build output filename: <subject>_ses-P28_Perilesional_Stroke_mask.nii.gz
    mapped_tp = session_mapping.get(timepoint, timepoint)
    out_fname = os.path.join(output_dir,
        f"{subject}_{mapped_tp}_Perilesional_Stroke_mask.nii.gz")
    nib.save(new_img, out_fname)
    print(f"Saved chronic perilesional mask: {out_fname}")
    return new_img, out_fname

# ----------------------------------------------------------------------------
# FUNCTION: apply_whole_brain_correction
# Overlaps the perilesional stroke mask with the whole-brain mask
# to ensure no voxels outside the brain remain.
# ----------------------------------------------------------------------------
def apply_whole_brain_correction(mask_path, anat_dir):
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata() > 0

    # Identify the brain-only mask (excludes stroke masks)
    brain_files = [f for f in glob.glob(os.path.join(anat_dir, "*BiasBet_mask.nii.gz"))
                   if "Stroke" not in f]
    if not brain_files:
        print(f"No whole brain mask found in {anat_dir} for correction.")
        return

    brain_img = nib.load(brain_files[0])
    brain_data = brain_img.get_fdata() > 0

    # Logical AND: keep only voxels inside both masks
    corrected = np.logical_and(mask_data, brain_data).astype(np.uint8)
    out_img = nib.Nifti1Image(corrected, mask_img.affine, mask_img.header)
    nib.save(out_img, mask_path)
    print(f"Applied whole brain correction to {mask_path}")

# ----------------------------------------------------------------------------
# FUNCTION: register_corrected_mask
# Registers a given perilesional mask into incidence space, then
# into every other timepoint for the subject, applying whole-brain
# correction at each stage.
# ----------------------------------------------------------------------------
def register_corrected_mask(corrected_mask_path, ses_anat_dir,
                            subject, session, subject_path, log_file_path):
    # Step 1: Resample to incidence space using inverse transform
    incidence_files = glob.glob(os.path.join(ses_anat_dir, "*IncidenceData.nii.gz"))
    inv_files       = glob.glob(os.path.join(ses_anat_dir, "*MatrixInv.txt"))
    if not incidence_files or not inv_files:
        with open(log_file_path, "a") as lf:
            lf.write(f"Incidence or inverse matrix missing in {ses_anat_dir}\n")
        return

    incidence = incidence_files[0]
    invmat    = inv_files[0]
    mapped_src = session_mapping.get(session, session)
    out_inc    = os.path.join(ses_anat_dir,
        f"{subject}_{mapped_src}_Perilesional_IncidenceSpace.nii.gz")

    # Call external reg_resample for incidence space
    os.system(
        f"reg_resample -ref {incidence} -flo {corrected_mask_path} "
        f"-inter 0 -trans {invmat} -res {out_inc}"
    )
    print(f"Created incidence space mask: {out_inc}")

    # Step 2: Register incidence-space mask to all other sessions
    for tp in os.listdir(subject_path):
        if tp == session:
            continue
        anat_tp = os.path.join(subject_path, tp, "anat")
        bspline_files = glob.glob(os.path.join(anat_tp, "*MatrixBspline.nii"))
        bet_files     = glob.glob(os.path.join(anat_tp, "*BiasBet.nii.gz"))
        if not bspline_files or not bet_files:
            with open(log_file_path, "a") as lf:
                lf.write(f"Missing registration files in {anat_tp}\n")
            continue

        out_tp = os.path.join(anat_tp,
            f"{subject}_{tp}_{mapped_src}_Perilesional_Stroke_mask.nii.gz"
        )
        os.system(
            f"reg_resample -ref {bet_files[0]} -flo {out_inc} "
            f"-inter 0 -trans {bspline_files[0]} -res {out_tp}"
        )
        print(f"Registered mask to {tp}: {out_tp}")

        # Apply whole-brain correction at each target timepoint
        apply_whole_brain_correction(out_tp, anat_tp)

# ----------------------------------------------------------------------------
# FUNCTION: global_whole_brain_correction
# Walks all subjects/sessions and applies whole-brain correction
# only to perilesional stroke masks.
# ----------------------------------------------------------------------------
def global_whole_brain_correction(input_path):
    subject_dirs = [d for d in glob.glob(os.path.join(input_path, "*"))
                    if os.path.isdir(d)]
    for subject_dir in subject_dirs:
        for sess in os.listdir(subject_dir):
            anat_dir = os.path.join(subject_dir, sess, "anat")
            if not os.path.isdir(anat_dir):
                continue

            # Glob only perilesional stroke masks
            perimasks = glob.glob(os.path.join(
                anat_dir, "*Perilesional_Stroke_mask.nii.gz"
            ))
            for mask_file in perimasks:
                apply_whole_brain_correction(mask_file, anat_dir)

# ----------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------
def main(input_path):
    log_file_path = "/tmp/missing_files_log.txt"

    # Step 1: Rename raw stroke masks per session mapping
    rename_stroke_masks(input_path)

    # Step 2: Acute phase perilesional masks (ses-P1 to ses-P5)
    acute_sessions = ["ses-P1", "ses-P2", "ses-P3",
                      "ses-P4", "ses-P5"]
    for subj_dir in sorted(glob.glob(os.path.join(input_path, "*"))):
        subject = os.path.basename(subj_dir)
        processed = False
        for sess in acute_sessions:
            anat_dir = os.path.join(subj_dir, sess, "anat")
            if not os.path.isdir(anat_dir):
                continue
            stroke_files = glob.glob(
                os.path.join(anat_dir, "*BiasBetStroke_mask.nii.gz")
            )
            if not stroke_files:
                continue

            print(f"Processing acute stroke mask: {stroke_files[0]}")
            peri_img = create_perilesional_mask_no_stroke(stroke_files[0])
            mapped_s = session_mapping[sess]
            out_name = os.path.join(
                anat_dir, f"{subject}_{mapped_s}_Perilesional_Stroke_mask.nii.gz"
            )
            nib.save(peri_img, out_name)

            # Register and correct
            register_corrected_mask(
                out_name, anat_dir, subject, sess,
                subj_dir, log_file_path
            )
            processed = True
            break

        if not processed:
            print(f"No acute stroke mask found for {subject}.")

    # Step 3: Chronic phase perilesional masks (e.g. ses-P27 to ses-P56)
    chronic_sessions = ["ses-P27", "ses-P28", "ses-P29",
                        "ses-P30", "ses-P56"]
    for subj_dir in sorted(glob.glob(os.path.join(input_path, "*"))):
        subject = os.path.basename(subj_dir)
        for sess in chronic_sessions:
            anat_dir = os.path.join(subj_dir, sess, "anat")
            if not os.path.isdir(anat_dir):
                continue
            pre_masks = glob.glob(
                os.path.join(anat_dir, "*_ses-P3_Stroke_mask.nii.gz")
            )
            if not pre_masks:
                print(f"No pre-registered stroke mask in {anat_dir} for {sess}.")
                continue

            img, out = create_perilesional_mask_sesP28(
                pre_masks[0], anat_dir, subject, sess, anat_dir
            )
            if img is None:
                continue

            register_corrected_mask(
                out, anat_dir, subject, sess,
                subj_dir, log_file_path
            )

    # Step 4: Global whole-brain correction on all perilesional masks
    print("Starting global whole brain correction on perilesional masks...")
    global_whole_brain_correction(input_path)
    print("Global whole brain correction complete.")

# Script entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and global-correct perilesional stroke masks across timepoints."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Path to the base directory containing subject/session/anat subfolders."
    )
    args = parser.parse_args()
    main(args.input)