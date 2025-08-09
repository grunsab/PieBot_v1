import os
import shutil

def organize_files():
    """
    Moves and renames files from test/train folders to the parent
    directory.
    """
    # Define the path to the main directory.
    # This script assumes it is run from the same directory that contains 'CCRL'.
    base_dir = 'cclr'

    # The source directory is now the same as the base directory.
    source_root = base_dir

    # The destination directory is also the base 'CCRL' folder.
    dest_dir = base_dir

    # List of subdirectories to process.
    subfolders_to_process = ['test', 'train']

    print("Starting file organization process...")

    # Ensure the destination directory exists.
    if not os.path.isdir(dest_dir):
        print(f"Error: Destination directory '{dest_dir}' not found.")
        return

    # Iterate over the 'test' and 'train' subfolders.
    for folder_name in subfolders_to_process:
        current_source_path = os.path.join(source_root, folder_name)

        # Check if the source subfolder exists before proceeding.
        if not os.path.isdir(current_source_path):
            print(f"Warning: Source folder '{current_source_path}' not found. Skipping.")
            continue

        # Get a list of all files in the current subfolder.
        try:
            files = os.listdir(current_source_path)
            print(f"\nFound {len(files)} files in '{current_source_path}'.")
        except OSError as e:
            print(f"Error reading directory {current_source_path}: {e}")
            continue

        # Process each file in the subfolder.
        for filename in files:
            # Construct the full path to the original file.
            old_filepath = os.path.join(current_source_path, filename)

            # Create the new filename with the appropriate prefix.
            # e.g., '1.pgn' in 'test' folder becomes 'test_1.pgn'
            new_filename = f"{folder_name}_{filename}"

            # Construct the full path for the new file location and name.
            new_filepath = os.path.join(dest_dir, new_filename)

            # Move the file from the old path to the new path.
            # The shutil.move() function handles both moving and renaming.
            try:
                shutil.move(old_filepath, new_filepath)
                print(f"  Moved: '{old_filepath}' -> '{new_filepath}'")
            except Exception as e:
                print(f"  Error moving file {old_filepath}: {e}")

    print("\nFile organization complete!")

if __name__ == "__main__":
    organize_files()

