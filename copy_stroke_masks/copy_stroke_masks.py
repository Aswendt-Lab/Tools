import os
import glob
import pandas as pd
import shutil

# Specify the input and output paths directly
input_path = r"E:\CRC_data\PRR\Stroke_masks"  # Replace with your actual input directory path
output_path = r"E:\CRC_data\PRR\proc_data"  # Replace with your actual output directory path

# Search for files matching "*Stroke_mask*" pattern in the input directory
temp_stroke_masks = glob.glob(os.path.join(input_path, "**/*Stroke_mask*"), recursive=True)

# Create a DataFrame for stroke masks
df_masks = pd.DataFrame({
    "exactAddress": temp_stroke_masks,
    "fileName": [os.path.basename(ss).replace(".nii.gz", "") for ss in temp_stroke_masks]
})

# Remove substrings matching "_*" from fileName using str.extract()
df_masks["fileName"] = df_masks["fileName"].str.extract(r'(.*?)_.*')

# Search for all subject folders in the output path and filter out non-folders
temp_subjects = [entry for entry in glob.glob(os.path.join(output_path, "*")) if os.path.isdir(entry)]

# Create a DataFrame for subjects
df_subjects = pd.DataFrame({
    "exactAddress": temp_subjects,
    "folderName": [os.path.basename(tt).replace("sub-", "") for tt in temp_subjects]
})

# Initialize a list to store move operations status
move_status = []

# Loop over unique filenames in df_masks
for mm in df_masks["fileName"].unique():
    if mm in df_subjects["folderName"].values:
        tempAd = df_subjects.loc[df_subjects["folderName"] == mm, "exactAddress"].values[0]
        TimePoints = glob.glob(os.path.join(tempAd, "*"))

        moved = False

        for time_point in ["ses-P1", "ses-P2", "ses-P3", "ses-P7"]:
            target_dirs = [tp for tp in TimePoints if time_point == os.path.basename(tp)]
            if target_dirs:
                for target_dir in target_dirs:
                    anat_dir = os.path.join(target_dir, "anat")
                    if not os.path.exists(anat_dir):
                        continue
                    shutil.move(df_masks[df_masks["fileName"]==mm]["exactAddress"].values[0], anat_dir)  # Use shutil.move() to move the file
                    move_status.append([df_masks[df_masks["fileName"]==mm]["exactAddress"].values[0], anat_dir, "Moved"])
                    moved = True
                    break
            if moved:
                break

        if not moved:
            move_status.append([tempAd, "Not Moved", "No relevant time point folders found"])

# Save move status to CSV
move_status_df = pd.DataFrame(move_status, columns=["Source", "Destination", "Status"])
move_status_df.to_csv(os.path.join(input_path, "move_status.csv"), index=False)

print("Data processing complete. The expanded data has been moved to", input_path)
