import os
import argparse
import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This script automates the renaming of files in a directory. It searches for files matching "*Stroke_mask*" in the specified input directory and renames them by replacing the first underscore with specific characters.')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to the parent project folder of the dataset, e.g., raw_data', type=str)
						
    ## Read out parameters
    args = parser.parse_args()

    # Search for files matching "*Stroke_mask*" pattern in the input directory
    subject_files = glob.glob(os.path.join(args.input, "*Stroke_mask*"), recursive=True)
    for ss in subject_files:
        # Extract the old file name and directory name
        old_file_name = os.path.basename(ss)
        directory_name = os.path.dirname(ss)
        
        # Create the new file name by replacing the first underscore with specific characters
        new_file_name = old_file_name.replace("_", "s", 1).replace("_", "c", 1).replace("_", "m", 1)
        
        # Construct the full path for the old and new file names
        old_file_path = os.path.join(directory_name, old_file_name)
        new_file_path = os.path.join(directory_name, new_file_name)
        
        # Rename the old files with new names
        os.rename(old_file_path, new_file_path)
        print(f"Renamed: {old_file_path} to {new_file_path}")

print("Data processing complete. Files have been renamed.")
