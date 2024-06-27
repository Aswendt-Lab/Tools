import os
import zipfile
import glob
import argparse

def zip_files_in_chunks(directory, output_directory, chunk_size_gb):
    # Convert chunk size to bytes
    chunk_size_bytes = chunk_size_gb * 1024 * 1024 * 1024
    current_size = 0
    part = 1
    temp_files = []

    # Create output directory for zip parts
    os.makedirs(output_directory, exist_ok=True)

    # Get all files in the directory recursively
    all_files = [f for f in glob.glob(directory + '/**', recursive=True) if os.path.isfile(f)]

    # Iterate over all files
    for file_path in all_files:
        file_size = os.path.getsize(file_path)

        # Check if adding the file would exceed the chunk size
        if current_size + file_size > chunk_size_bytes:
            # Create a zip file for the current batch of files
            zip_filename = os.path.join(output_directory, f'archive_part_{part}.zip')
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file in temp_files:
                    arcname = os.path.relpath(file, directory)
                    zipf.write(file, arcname)

            # Increment part number and reset size counter and temp files list
            part += 1
            current_size = 0
            temp_files = []

        # Add file to temp list and update the current size
        temp_files.append(file_path)
        current_size += file_size

    # Zip the remaining files if any
    if temp_files:
        zip_filename = os.path.join(output_directory, f'archive_part_{part}.zip')
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in temp_files:
                arcname = os.path.relpath(file, directory)
                zipf.write(file, arcname)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zip files into equally sized chunks.')
    parser.add_argument('-i', '--input_directory', type=str, required=True, help='Directory containing files to zip.')
    parser.add_argument('-o', '--output_directory', type=str, required=True, help='Directory to save zip parts.')
    parser.add_argument('-c', '--chunk_size', type=int, default=5, help='Maximum size of each zip part in GB.')
    
    args = parser.parse_args()
    
    zip_files_in_chunks(args.input_directory, args.output_directory, args.chunk_size)
