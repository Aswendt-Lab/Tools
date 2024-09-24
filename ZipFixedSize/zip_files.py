import os
import zipfile
import glob
import argparse
from tqdm import tqdm

def zip_files_in_chunks(directory, output_directory, chunk_size_gb, keep_structure):
    """
    Compress files from the specified directory into multiple zip files of a defined maximum size.
    
    Parameters:
    directory (str): Path to the directory containing files to zip.
    output_directory (str): Path where the zip files will be saved.
    chunk_size_gb (int): Maximum size of each zip file (in GB).
    keep_structure (bool): Flag to preserve the folder structure in the zip file or flatten it.
    
    Returns:
    None
    """
    # Convert chunk size from GB to bytes
    chunk_size_bytes = chunk_size_gb * 1024 * 1024 * 1024
    current_size = 0
    part = 1
    temp_files = []

    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Get all files in the directory recursively, excluding existing .zip files
    all_files = [f for f in glob.glob(directory + '/**', recursive=True) if os.path.isfile(f) and not f.endswith('.zip')]

    if not all_files:
        print("No files found in the specified directory.")
        return

    # Initialize the progress bar
    with tqdm(total=len(all_files), desc="Zipping files", unit="file") as pbar:
        for file_path in all_files:
            file_size = os.path.getsize(file_path)

            # Check if adding this file would exceed the chunk size
            if current_size + file_size > chunk_size_bytes:
                zip_filename = os.path.join(output_directory, f'archive_part_{part}.zip')
                
                # Write the current batch of files to the zip
                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file in temp_files:
                        if keep_structure:
                            arcname = os.path.relpath(file, directory)
                        else:
                            arcname = os.path.basename(file)
                        zipf.write(file, arcname)

                part += 1  # Move to the next part
                current_size = 0  # Reset size counter
                temp_files = []  # Clear the temporary file list

            temp_files.append(file_path)
            current_size += file_size
            pbar.update(1)  # Update the progress bar after processing each file

        # Handle any remaining files
        if temp_files:
            zip_filename = os.path.join(output_directory, f'archive_part_{part}.zip')
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in temp_files:
                    if keep_structure:
                        arcname = os.path.relpath(file, directory)
                    else:
                        arcname = os.path.basename(file)
                    zipf.write(file, arcname)

    print(f"Zipping complete. {part} zip files created in '{output_directory}'.")

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(
        description=(
            "Compress files in a directory into multiple zip files, each with a maximum size."
            " Useful for archiving large datasets or logs while controlling the size of each zip."
        ),
        epilog="Example: python zip_in_chunks.py -i /path/to/input -o /path/to/output -c 5 -k"
    )
    parser.add_argument(
        '-i', '--input_directory', 
        type=str, 
        required=True, 
        help='Directory containing the files to compress.'
    )
    parser.add_argument(
        '-o', '--output_directory', 
        type=str, 
        required=True, 
        help='Directory where the resulting zip files will be saved. This can be the same as the input directory.'
    )
    parser.add_argument(
        '-c', '--chunk_size', 
        type=int, 
        default=5, 
        help='Maximum size (in GB) of each zip file. Default is 5 GB.'
    )
    parser.add_argument(
        '-k', '--keep_structure', 
        action='store_true', 
        help='Preserve the folder structure inside the zip file. If not set, all files are zipped flat.'
    )
    
    # Parse arguments
    args = parser.parse_args()

    # Ensure the input directory exists
    if not os.path.isdir(args.input_directory):
        print(f"Error: Input directory '{args.input_directory}' does not exist.")
    else:
        # Call the function to zip files
        zip_files_in_chunks(args.input_directory, args.output_directory, args.chunk_size, args.keep_structure)
