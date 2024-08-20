import os
import glob
import pandas as pd
import argparse

def find_paths_without_method(input_path):
    # Construct the search pattern for acqp files
    path = os.path.join(input_path, "**", "*acqp")
    listA = glob.glob(path, recursive=True)

    # Initialize an empty list to hold paths that don't have a method file
    no_method_paths = []

    # Loop through the list of acqp files
    for mm in listA:
        temp = os.path.dirname(mm)
        method_file = os.path.join(temp, "method")
        method_exists = glob.glob(method_file)

        if not method_exists:  # Check if method_exists is empty
            no_method_paths.append(temp)

    return no_method_paths

def find_corrupted_subject_files(input_path):
    # Construct the search pattern for subject files
    path = os.path.join(input_path, "**", "subject")
    listB = glob.glob(path, recursive=True)

    # Initialize an empty list to hold paths with corrupted subject files
    corrupted_subject_paths = []

    # Loop through the list of subject files
    for subject_file in listB:
        try:
            with open(subject_file, 'r') as file:
                content = file.read()
                if "##OWNER=" not in content or "##OWNER=\n" in content:
                    corrupted_subject_paths.append(os.path.dirname(subject_file))
        except Exception as e:
            corrupted_subject_paths.append(os.path.dirname(subject_file))

    return corrupted_subject_paths

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Find directories without method files and check for corrupted subject files.')
    parser.add_argument('-i', '--inputpath', required=True, help='Initial input path')
    parser.add_argument('-o', '--outputpath', required=True, help='Output path for the CSV file')

    # Parse arguments
    args = parser.parse_args()

    # Find paths without method files
    no_method_paths = find_paths_without_method(args.inputpath)
    # Find paths with corrupted subject files
    corrupted_subject_paths = find_corrupted_subject_files(args.inputpath)

    # Save the list of paths to a CSV file
    output_csv = args.outputpath

    # Create a DataFrame for paths without method files
    df_no_method = pd.DataFrame(no_method_paths, columns=["Path_Without_Method"])
    # Create a DataFrame for corrupted subject paths
    df_corrupted_subject = pd.DataFrame(corrupted_subject_paths, columns=["Corrupted_Subject_Path"])

    # Concatenate DataFrames
    df = pd.concat([df_no_method, df_corrupted_subject], axis=1)
    df.to_csv(output_csv, index=False)

    print(f"Paths without a method file and corrupted subject files have been saved to {output_csv}")

if __name__ == "__main__":
    main()
