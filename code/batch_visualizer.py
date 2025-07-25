"""
ZInD Batch Visualization Tool

This script finds and visualizes all previously generated .npy grid map files
within a specified root directory, calling the `visualize_zind_cli.py` tool
to save a JPEG image for each one.

It assumes a directory structure where .npy files exist, for example:
- [root_dir]/
  - 0000/
    - zind_data.json
    - map_res0.05_raw_merged.npy  <-- INPUT
    - map_res0.05_raw_merged.meta.json
  - 0001/
    - zind_data.json
    - map_res0.05_raw_merged.npy  <-- INPUT
    - map_res0.05_raw_merged.meta.json
  - ...

The output for each scene (the .jpeg file) will be saved within its
corresponding scene directory (e.g., in `0000/`).
"""
import sys
import argparse
import subprocess
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser(
        description="ZInD Batch Visualization Tool.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "root_directory",
        type=str,
        help="The root directory containing the scene subdirectories with .npy files."
    )
    parser.add_argument(
        "tool_script_path",
        type=str,
        help="The path to the visualization script (e.g., 'visualize_zind_cli.py')."
    )
    parser.add_argument(
        "--npy_pattern",
        type=str,
        default="map_*.npy",
        help="The pattern to search for .npy files recursively (default: 'map_*.npy')."
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="Force re-visualization and overwrite existing output JPEG files. Default is to skip."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would be executed without running them."
    )

    par
    args = parser.parse_args()

    # --- Validate Paths ---
    root_dir = Path(args.root_directory)
    tool_script = Path(args.tool_script_path)

    if not root_dir.is_dir():
        print(f"Error: Root directory not found at '{root_dir.resolve()}'")
        sys.exit(1)
    if not tool_script.is_file():
        print(f"Error: Tool script not found at '{tool_script.resolve()}'")
        sys.exit(1)

    # --- Find all .npy files recursively ---
    print(f"Searching for '{args.npy_pattern}' files in '{root_dir}'...")
    npy_files = sorted(list(root_dir.rglob(args.npy_pattern)))
    total_files = len(npy_files)
    print(f"Found {total_files} .npy files to visualize.\n")

    if total_files == 0:
        print("No files found to process. Exiting.")
        return

    success_count = 0
    failure_count = 0
    skip_count = 0
    start_time = time.time()

    # --- Main Visualization Loop ---
    for i, npy_path in enumerate(npy_files):
        scene_dir = npy_path.parent
        print(f"--- Visualizing {i+1}/{total_files}: {npy_path.name} in {scene_dir.name} ---")

        # Define the expected output path
        output_jpeg_path = scene_dir / f"{npy_path.stem}.jpeg"

        # --- Check for existing files ---
        if output_jpeg_path.exists() and not args.force_overwrite and not args.dry_run:
            print(f"  [SKIP] Output file already exists: {output_jpeg_path.name}")
            print("-" * (len(npy_path.name) + 25) + "\n")
            skip_count += 1
            continue

        # --- Construct the command to execute ---
        command = [
            sys.executable,
            str(tool_script),
            "--visualize_npy", str(npy_path),
            "--output", str(scene_dir) # Output is the parent directory of the npy
        ]

        # --- Execute or Dry Run ---
        if args.dry_run:
            print(f"  [DRY RUN] Would execute: {' '.join(command)}")
            continue

        try:
            print(f"  Executing visualization...")
            result = subprocess.run(
                command,
                check=True,
                timeout=60
            )
            print(f"  [SUCCESS] Visualization saved to {output_jpeg_path.name}")
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Visualization tool failed on {npy_path.name} with exit code {e.returncode}.")
            print("  ---------- FAILED TOOL STDERR ----------")
            print(e.stderr)
            print("  --------------------------------------")
            failure_count += 1
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Tool timed out on {npy_path.name}.")
            failure_count += 1
        except Exception as e:
            print(f"  [CRITICAL ERROR] An unexpected error occurred while processing {npy_path.name}: {e}")
            failure_count += 1
        
        print("-" * (len(npy_path.name) + 25) + "\n")

    # --- Final Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    print("="*40)
    print("Batch Visualization Complete")
    print("="*40)
    print(f"Total .npy files found: {total_files}")
    print(f"  Visualized successfully: {success_count}")
    print(f"  Skipped (already exist): {skip_count}")
    print(f"  Failures:                {failure_count}")
    print(f"Total time elapsed: {total_time:.2f} seconds")
    print("="*40)

if __name__ == '__main__':
    main()