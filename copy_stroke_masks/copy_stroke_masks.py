import os
import argparse
import glob
import pandas as pd
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script automates the renaming of files in a directory. It searches for files matching "*Stroke_mask*" in the specified input directory and renames them by replacing the first underscore with specific characters.')
    parser.add_argument('-i', '--input', required=True, help='Path to the parent project folder of the dataset, e.g., raw_data', type=str)
    parser.add_argument('-o', '--output', required=True, help='Path to the output folder, e.g., proc_data', type=str)
    
    # Read out parameters
    args = parser.parse_args()

    # Search for files matching "*Stroke_mask*" pattern in the input directory
    temp_stroke_masks = glob.glob(os.path.join(args.input, "**/*Stroke_mask*"), recursive=True)
    
    # Create a DataFrame for stroke masks
    df_masks = pd.DataFrame({
        "exactAddress": temp_stroke_masks,
        "fileName": [os.path.basename(ss).replace(".nii.gz", "") for ss in temp_stroke_masks]
    })

    # Search for all subject folders in the output path
    temp_subjects = glob.glob(os.path.join(args.output, "*"))
    
    # Create a DataFrame for subjects
    df_subjects = pd.DataFrame({
        "exactAddress": temp_subjects,
        "folderName": [os.path.basename(tt).replace("sub-", "") for tt in temp_subjects]
    })
    
    # Initialize a list to store copy operations status
    copy_status = []

    # Loop over unique filenames in df_masks
    for mm in df_masks["fileName"].unique():
        if mm in df_subjects["folderName"].values:
            tempAd = df_masks.loc[df_masks["fileName"] == mm, "exactAddress"].values[0]
            TimePoints = glob.glob(os.path.join(tempAd, "*"))

            copied = False
            reasons = []

            for time_point in ["P1", "P2", "P3", "P7"]:
                target_dirs = [tp for tp in TimePoints if time_point in os.path.basename(tp)]
                if target_dirs:
                    for target_dir in target_dirs:
                        anat_dir = os.path.join(target_dir, "anat")
                        if not os.path.exists(anat_dir):
                            os.makedirs(anat_dir)
                        shutil.copy(tempAd, anat_dir)
                        copy_status.append([tempAd, anat_dir, "Copied"])
                        copied = True
                        break
                if copied:
                    break

            if not copied:
                copy_status.append([tempAd, "Not Copied", "No relevant time point folders found"])

    # Save copy status to CSV
    copy_status_df = pd.DataFrame(copy_status, columns=["Source", "Destination", "Status"])
    copy_status_df.to_csv(os.path.join(args.input, "copy_status.csv"), index=False)

    print("Data processing complete. The expanded data has been saved to", args.input)
