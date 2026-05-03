# Motahare, 02.04.25, Generating the Contralesional Stroke_mask for Stroke_mask 

import os
import glob
import argparse
import subprocess
import shlex
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

# Automatically get NiftyReg directory from the environment variable NIFTYREGDIR.
# If defined, we use its bin folder; otherwise, we assume the commands are on PATH.
NIFTYREGDIR = os.environ.get("NIFTYREGDIR")
if NIFTYREGDIR:
    NIFTY_BIN = os.path.join(NIFTYREGDIR, "bin") + os.sep
    print(f"Using NiftyReg from NIFTYREGDIR: {NIFTY_BIN}")
else:
    NIFTY_BIN = ""
    print("NIFTYREGDIR environment variable not found. Using system PATH for NiftyReg commands.")

def register_to_template(native_image, template, output_image, out_mat):
    """
    Registers the native T2wBiasBet image to the in-house template using reg_aladin.
    """
    command = f"{NIFTY_BIN}reg_aladin -ref {template} -flo {native_image} -res {output_image} -aff {out_mat}"
    print(f"Running registration command: {command}")
    subprocess.run(shlex.split(command), check=True)
    print(f"Registration complete. Registered image saved as: {output_image}")

def invert_transformation_matrix(mat_file, inv_mat_file):
    """
    Inverts the affine transformation matrix using NumPy.
    Reads the matrix from mat_file, computes its inverse, and saves it to inv_mat_file.
    """
    print("Inverting transformation matrix using numpy...")
    try:
        mat = np.loadtxt(mat_file)
        inv_mat = np.linalg.inv(mat)
        np.savetxt(inv_mat_file, inv_mat, fmt="%.6f")
        print(f"Inverted matrix saved as: {inv_mat_file}")
    except Exception as e:
        print(f"Error inverting matrix: {e}")
        raise

def compute_midline_from_template(registered_image):
    """
    Loads the registered image (now in template space) and computes the midline.
    Assumes that the x-axis is left–right.
    Uses a representative coronal slice (middle of the y-dimension) and computes
    the median x-coordinate of nonzero voxels.
    """
    img = nib.load(registered_image)
    data = img.get_fdata()
    nx, ny, nz = data.shape
    y_mid = ny // 2
    # Extract a coronal slice (all x and z, at y_mid)
    coronal_slice = data[:, y_mid, :]
    
    # Identify rows (x-indices) with nonzero voxels
    x_indices = np.where(coronal_slice.sum(axis=1) > 0)[0]
    if len(x_indices) == 0:
        print("Warning: No nonzero voxels found in the representative coronal slice. Using image center.")
        mid_x = nx // 2
    else:
        mid_x = int(np.median(x_indices))
    print(f"Computed midline (x coordinate in template space): {mid_x} (Image width: {nx})")
    return mid_x

def flip_mask_along_midline(registered_mask, mid_x, output_flipped_mask):
    """
    Flips the stroke mask along the x-axis using the computed midline.
    Loads the stroke mask (assumed to be in template space) and mirrors voxels from
    the ipsilesional hemisphere (determined by voxel count) across the midline.
    """
    img = nib.load(registered_mask)
    data = img.get_fdata()
    mask = (data > 0).astype(np.uint8)
    nx, ny, nz = mask.shape

    # Determine ipsilesional side by comparing voxel counts on each side of midline.
    left_voxels = np.sum(mask[:mid_x, :, :])
    right_voxels = np.sum(mask[mid_x:, :, :])
    
    if left_voxels >= right_voxels:
        ipsi_side = 'left'
        print("Assuming ipsilesional stroke is on the left hemisphere.")
    else:
        ipsi_side = 'right'
        print("Assuming ipsilesional stroke is on the right hemisphere.")

    mirror_mask = np.zeros_like(mask)
    if ipsi_side == 'left':
        for x in range(mid_x):
            mirrored_x = 2 * mid_x - x
            if mirrored_x < nx:
                mirror_mask[mirrored_x, :, :] = np.logical_or(mirror_mask[mirrored_x, :, :], mask[x, :, :])
    else:
        for x in range(mid_x, nx):
            mirrored_x = 2 * mid_x - x
            if 0 <= mirrored_x < nx:
                mirror_mask[mirrored_x, :, :] = np.logical_or(mirror_mask[mirrored_x, :, :], mask[x, :, :])
    
    new_img = nib.Nifti1Image(mirror_mask.astype(np.uint8), img.affine, img.header)
    nib.save(new_img, output_flipped_mask)
    print(f"Flipped (contralesional) mask saved as: {output_flipped_mask}")

def transform_mask_back(native_image, flipped_mask, inv_mat, output_native_mask):
    """
    Transforms the flipped mask (in template space) back to native space using reg_resample
    with the inverse transformation matrix.
    """
    command = f"{NIFTY_BIN}reg_resample -ref {native_image} -flo {flipped_mask} -trans {inv_mat} -res {output_native_mask}"
    print(f"Transforming flipped mask back to native space with command: {command}")
    subprocess.run(shlex.split(command), check=True)
    print(f"Transformed mask saved as: {output_native_mask}")

def apply_whole_brain_correction(mask_path, anat_dir):
    """
    After generation, applies whole brain correction by loading the whole brain mask 
    (file matching '*BiasBet_mask.nii.gz' excluding 'Stroke') in anat_dir and removing voxels
    outside the brain. The corrected mask overwrites the original mask.
    """
    mask_img = nib.load(mask_path)
    mask_data = mask_img.get_fdata() > 0
    brain_mask_pattern = os.path.join(anat_dir, "*BiasBet_mask.nii.gz")
    brain_mask_files = [f for f in glob.glob(brain_mask_pattern) if "Stroke" not in f]
    if not brain_mask_files:
        print(f"No whole brain mask found in {anat_dir} for correction.")
        return
    brain_img = nib.load(brain_mask_files[0])
    brain_data = brain_img.get_fdata() > 0
    corrected_data = np.logical_and(mask_data, brain_data).astype(np.uint8)
    new_img = nib.Nifti1Image(corrected_data, mask_img.affine, mask_img.header)
    nib.save(new_img, mask_path)
    print(f"Applied whole brain correction to {mask_path}")

def process_anat_folder(anat_folder, template):
    """
    For a given anat folder:
      1. Find the native T2wBiasBet image (*BiasBet.nii.gz).
      2. Register it to the in-house template (using reg_aladin) and compute the midline.
      3. For each stroke mask (*Stroke_mask.nii.gz) in the folder:
            a. Transform it to template space using reg_resample.
            b. Flip it along the computed midline.
            c. Transform the flipped mask back to native space using reg_resample.
         The new mask is saved with the suffix "ContraStroke_mask.nii.gz".
         Finally, apply whole brain correction on the generated ContraStroke_mask.
    """
    print(f"\nProcessing folder: {anat_folder}")
    biasbet_files = glob.glob(os.path.join(anat_folder, "*BiasBet.nii.gz"))
    if not biasbet_files:
        print(f"No BiasBet image found in {anat_folder}. Skipping...")
        return
    native_t2 = biasbet_files[0]
    print(f"Using T2 image: {native_t2}")

    base_t2 = os.path.basename(native_t2).split('.')[0]
    reg_t2 = os.path.join(anat_folder, base_t2 + '_regT2.nii.gz')
    trans_mat = os.path.join(anat_folder, base_t2 + '_xfm.txt')
    inv_mat = os.path.join(anat_folder, base_t2 + '_xfm_inv.txt')

    register_to_template(native_t2, template, reg_t2, trans_mat)
    invert_transformation_matrix(trans_mat, inv_mat)
    midline_x = compute_midline_from_template(reg_t2)

    stroke_mask_files = glob.glob(os.path.join(anat_folder, "*Stroke_mask.nii.gz"))
    if not stroke_mask_files:
        print(f"No stroke mask found in {anat_folder}.")
        return

    for stroke_mask in stroke_mask_files:
        print(f"\nProcessing stroke mask: {stroke_mask}")
        base_mask = os.path.basename(stroke_mask).replace(".nii.gz", "")
        reg_stroke_mask = os.path.join(anat_folder, base_mask + '_reg.nii.gz')
        command = f"{NIFTY_BIN}reg_resample -ref {template} -flo {stroke_mask} -trans {trans_mat} -res {reg_stroke_mask}"
        print(f"Transforming stroke mask to template space with command: {command}")
        subprocess.run(shlex.split(command), check=True)
        print(f"Registered stroke mask saved as: {reg_stroke_mask}")

        flipped_mask = os.path.join(anat_folder, base_mask + '_flippedStroke_mask.nii.gz')
        flip_mask_along_midline(reg_stroke_mask, midline_x, flipped_mask)

        contra_mask = os.path.join(anat_folder, base_mask.replace("Stroke_mask", "ContraStroke_mask") + ".nii.gz")
        transform_mask_back(native_t2, flipped_mask, inv_mat, contra_mask)
        print(f"Contra mask created: {contra_mask}")
        # Apply whole brain correction to the generated ContraStroke_mask.
        apply_whole_brain_correction(contra_mask, anat_folder)

def main(main_folder, template):
    """
    Loops over all subject directories under main_folder.
    In each subject folder, for each timepoint, finds the 'anat' folder,
    then processes that folder to register the T2 image, compute the midline,
    and flip stroke masks to produce ContraStroke_mask.nii.gz.
    Whole brain correction is applied to each generated ContraStroke_mask.
    """
    anat_folders = glob.glob(os.path.join(main_folder, "**", "anat"), recursive=True)
    if not anat_folders:
        print("No 'anat' folders found in the specified main folder.")
        return

    print(f"Found {len(anat_folders)} 'anat' folders. Starting processing...\n")
    for anat_folder in anat_folders:
        try:
            process_anat_folder(anat_folder, template)
        except Exception as e:
            print(f"Error processing folder {anat_folder}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process mouse brain data: "
                    "For each subject and timepoint in the main folder, "
                    "find the 'anat' folder, register the T2wBiasBet image to the template using reg_aladin, "
                    "compute the midline, flip stroke masks (ending with 'Stroke_mask.nii.gz') to produce "
                    "contralesional masks (with 'ContraStroke_mask.nii.gz'), and apply whole brain correction "
                    "using the BiasBet_mask in the same anat folder."
    )
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to the main folder containing subject subfolders")
    parser.add_argument("-tpl", "--template", type=str, required=True,
                        help="Path to the in-house template (e.g. NP_template_sc0.nii.gz)")
    args = parser.parse_args()
    main(args.input, args.template)