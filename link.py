"""
ZInD Symlink View Creator

This script creates an alternative "view" of a processed Zillow Indoor Dataset
directory. It scans a source directory for generated artifacts (like .npy maps,
.meta.json files, and .jpeg visualizations) and creates a new directory structure
where artifacts are grouped by type, using symbolic links to point to the
original files.

This allows for easy navigation and bulk operations on specific types of
generated data without duplicating any files.

Example:
- Original file: `data/0001/map_res0.05_raw.npy`
- Is linked from: `data_view/map_res0.05_raw_npy/0001.npy`

Usage:
    python create_symlink_view.py [source_data_dir] [destination_view_dir]
"""

import sys
import argparse
import os
import shutil
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(
        description="Create a symbolic link view of a processed ZInD directory.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "source_directory",
        type=str,
        help="The root directory containing the original scene subdirectories (e.g., 'data/')."
    )
    parser.add_argument(
        "view_directory",
        type=str,
        help="The destination directory where the symlink view will be created (e.g., 'data_view/')."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="If the view directory already exists, delete it completely before starting. Use with caution."
    )
    parser.add_argument(
        "--exclude",
        nargs='+',
        default=["zind_data.json"],
        help="A list of filenames to exclude from the view (default: 'zind_data.json')."
    )
    args = parser.parse_args()

    # --- Validate Paths ---
    source_dir = Path(args.source_directory).resolve()
    view_dir = Path(args.view_directory).resolve()

    if not source_dir.is_dir():
        print(f"Error: Source directory not found at '{source_dir}'")
        sys.exit(1)

    if view_dir.exists():
        if args.clean:
            print(f"Cleaning existing view directory: {view_dir}")
            try:
                shutil.rmtree(view_dir)
            except OSError as e:
                print(f"Error: Could not remove directory {view_dir}. Please check permissions.")
                print(e)
                sys.exit(1)
        else:
            print(f"Error: View directory '{view_dir}' already exists.")
            print("Use the --clean flag to automatically remove it before running.")
            sys.exit(1)
    
    try:
        view_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create directory {view_dir}. Please check permissions.")
        print(e)
        sys.exit(1)

    # --- Find all artifact files ---
    print(f"Scanning '{source_dir}' for artifact files...")
    # We find all files in the subdirectories
    all_files = [f for f in source_dir.rglob('*') if f.is_file() and f.name not in args.exclude]
    
    if not all_files:
        print("No artifact files found to create a view from. Exiting.")
        sys.exit(0)

    print(f"Found {len(all_files)} artifacts. Creating symlink view at '{view_dir}'...")

    link_count = 0
    error_count = 0

    # --- Main Symlinking Loop ---
    for original_path in tqdm(all_files, desc="Creating symlinks"):
        try:
            # Deconstruct the original path
            # e.g., /path/to/data/0001/map_res0.05_raw.npy
            scene_id = original_path.parent.name # -> "0001"
            artifact_filename = original_path.name   # -> "map_res0.05_raw.npy"
            
            # Construct the new view paths
            # Replace dots in filename with underscores for a clean directory name
            artifact_view_dir_name = artifact_filename.replace('.', '_')
            artifact_view_dir = view_dir / artifact_view_dir_name # -> .../data_view/map_res0.05_raw_npy/
            
            # The new link will be named after the scene ID, with the original extension
            symlink_name = f"{scene_id}{original_path.suffix}" # -> "0001.npy"
            symlink_path = artifact_view_dir / symlink_name # -> .../data_view/map_res0.05_raw_npy/0001.npy
            
            # Ensure the artifact-specific subdirectory exists
            artifact_view_dir.mkdir(parents=True, exist_ok=True)
            
            # --- Create the relative symbolic link ---
            # This is crucial for portability. The link target is relative to the link's location.
            target_path_relative = os.path.relpath(original_path, start=artifact_view_dir)
            
            if not symlink_path.exists():
                symlink_path.symlink_to(target_path_relative)
                link_count += 1

        except Exception as e:
            print(f"\nCould not process file {original_path}: {e}")
            error_count += 1
    
    # --- Final Summary ---
    print("\n" + "="*40)
    print("Symlink View Creation Complete")
    print("="*40)
    print(f"  Total symbolic links created: {link_count}")
    if error_count > 0:
        print(f"  Errors encountered: {error_count}")
    print(f"  View directory: {view_dir}")
    print("="*40)


if __name__ == "__main__":
    main()