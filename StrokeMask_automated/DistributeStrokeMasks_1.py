import os
import glob
import argparse

def main(inputPath):
    log_file_path = "/tmp/missing_files_log.txt"
    # Search for all stroke masks recursively in the "anat" folders.
    search_path = os.path.join(inputPath, "**", "anat", "*Stroke_mask.nii.gz")
    stroke_mask_files = glob.glob(search_path, recursive=True)
    print("Found stroke masks:", stroke_mask_files)
    
    for stroke_mask in stroke_mask_files:
        # Assuming the directory structure: <inputPath>/<Subject>/<timepoint>/anat/<stroke_mask>
        path_parts = stroke_mask.split(os.sep)
        subject = path_parts[-4]
        source_timepoint = path_parts[-3]
        
        try:
            # Look for the incidence image and inverse matrix in the same folder as the stroke mask.
            incidence_path = glob.glob(os.path.join(os.path.dirname(stroke_mask), "*IncidenceData.nii.gz"))[0]
            matrix_inv = glob.glob(os.path.join(os.path.dirname(stroke_mask), "*MatrixInv.txt"))[0]
        except IndexError:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"TransMatInv or IncidenceData not found for: {stroke_mask}\n")
            continue

        # First, register the stroke mask to the incidence space.
        output_stroke_incidence = os.path.join(os.path.dirname(stroke_mask),
                                                 f"{subject}_{source_timepoint}_StrokeM_IncidenceSpace.nii.gz")
        command1 = (f"reg_resample -ref {incidence_path} -flo {stroke_mask} -inter 0 "
                    f"-trans {matrix_inv} -res {output_stroke_incidence}")
        os.system(command1)
        
        # Find all timepoints for the subject.
        subject_path = os.path.join(inputPath, subject)
        timepoint_dirs = glob.glob(os.path.join(subject_path, "*"))
        
        for tp in timepoint_dirs:
            # Do not process the timepoint where the stroke mask originates.
            if os.path.basename(tp) != source_timepoint:
                anat_dir = os.path.join(tp, "anat")
                # Locate the necessary registration files in this destination timepoint.
                affine_glob = glob.glob(os.path.join(anat_dir, "*MatrixAff.txt"))
                bspline_glob = glob.glob(os.path.join(anat_dir, "*MatrixBspline.nii"))
                bet_file_glob = glob.glob(os.path.join(anat_dir, "*BiasBet.nii.gz"))
                
                if not affine_glob:
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Affine matrix file not found for: {tp}\n")
                    continue
                
                if not bspline_glob:
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"Bspline matrix file not found for: {tp}\n")
                    continue
                
                if not bet_file_glob:
                    with open(log_file_path, "a") as log_file:
                        log_file.write(f"BiasBet file not found for: {tp}\n")
                    continue
                
                # Use the first found files.
                matrix_aff = affine_glob[0]
                matrix_bspline = bspline_glob[0]
                bet_file = bet_file_glob[0]
                
                # Save the registered stroke mask with the new naming scheme:
                # {subjectID}_{sourceTimepoint}_Stroke_mask.nii.gz
                output_stroke = os.path.join(anat_dir, f"{subject}_{source_timepoint}_Stroke_mask.nii.gz")
                command2 = (f"reg_resample -ref {bet_file} -flo {output_stroke_incidence} -inter 0 "
                            f"-trans {matrix_bspline} -res {output_stroke}")
                os.system(command2)
                print(f"Registered stroke mask from {source_timepoint} to {os.path.basename(tp)} "
                      f"as {output_stroke}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and register stroke mask files across timepoints.")
    parser.add_argument("-i", "--input", type=str, help="Input path", required=True)
    args = parser.parse_args()

    main(args.input)