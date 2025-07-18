"""
ZInD Batch Processing Tool

This script finds and processes all Zillow Indoor Dataset JSON files within a
specified root directory, calling the main generation tool for each one.

It assumes a directory structure like:
- [root_dir]/
  - 0000/
    - zind_data.json
  - 0001/
    - zind_data.json
  - ...

The output for each scene (e.g., the .npy and .meta.json files) will be saved
within its corresponding scene directory (e.g., in `0000/`).

This script is designed to be robust, logging errors for individual files
without halting the entire batch process.
"""
import sys
import argparse
import subprocess
from pathlib import Path
import time

def main():
    parser = argparse.ArgumentParser(
        description="ZInD Batch Processing Tool.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "root_directory",
        type=str,
        help="The root directory containing the scene subdirectories (e.g., 'data/')."
    )
    parser.add_argument(
        "tool_script_path",
        type=str,
        help="The path to the main grid map generation script (e.g., 'create_zind_map.py')."
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="zind_data.json",
        help="The name of the JSON file to find in each scene directory (default: 'zind_data.json')."
    )
    
    # --- Passthrough arguments for the main tool ---
    parser.add_argument(
        "-r", "--resolution",
        type=float,
        default=0.04,
        help="Grid resolution in meters/pixel to pass to the tool."
    )
    parser.add_argument(
        "--subdivision_method",
        type=str,
        choices=['none', 'raw', 'voronoi'],
        default='raw',
        help="Subdivision algorithm to pass to the tool."
    )
    parser.add_argument(
        "--merge_unlabeled",
        action="store_true",
        help="Pass the '--merge_unlabeled' flag to the tool."
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands that would be executed without running them."
    )

    parser.add_argument("--viz_vector", action="store_true", help="Show/save a plot of the intermediary vector data for debugging.")

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

    # --- Find all scene directories ---
    # We sort them to ensure a consistent processing order
    scene_dirs = sorted([d for d in root_dir.glob('*') if d.is_dir()])
    total_scenes = len(scene_dirs)
    print(f"Found {total_scenes} potential scene directories in '{root_dir}'.\n")

    success_count = 0
    failure_count = 0
    start_time = time.time()

    # --- Main Processing Loop ---
    for i, scene_dir in enumerate(scene_dirs):
        print(f"--- Processing scene {i+1}/{total_scenes}: {scene_dir.name} ---")
        
        input_json_path = scene_dir / args.filename
        if not input_json_path.is_file():
            print(f"  [SKIP] JSON file '{args.filename}' not found in this directory.")
            continue

        # Define the output path inside the scene directory
        # Name the output based on the method used for clarity
        output_base_name = f"map_res{args.resolution}_{args.subdivision_method}"
        if args.merge_unlabeled:
            output_base_name += "_merged"
        output_npy_path = scene_dir / f"{output_base_name}.npy"

        # --- Construct the command to execute ---
        command = [
            sys.executable,  # Use the same python interpreter that is running this script
            str(tool_script),
            str(input_json_path),
            "--output_path", str(output_npy_path),
            "--resolution", str(args.resolution),
            "--subdivision_method", args.subdivision_method,
            "--no_viz", # Always suppress visualization in batch mode
            "--save_visualizations"

        ]
        if args.merge_unlabeled:
            command.append("--merge_unlabeled")
        if args.viz_vector:
            command.append("--viz_vector")
            
        # --- Execute or Dry Run ---
        if args.dry_run:
            print(f"  [DRY RUN] Would execute: {' '.join(command)}")
            success_count += 1
            continue

        try:
            print(f"  Executing...")
            result = subprocess.run(
                command,
                check=True,         # Raise an exception if the tool exits with an error       # Decode output as text
                timeout=300         # Add a timeout (in seconds) for very complex scenes
            )
            print(f"  [SUCCESS] Output saved to {output_npy_path}")
            # print(f"  Tool Output:\n{result.stdout}") # Uncomment for verbose logging
            success_count += 1

        except subprocess.CalledProcessError as e:
            print(f"  [ERROR] Tool failed on scene {scene_dir.name} with exit code {e.returncode}.")
            print("  ---------- FAILED TOOL STDERR ----------")
            print(e.stderr)
            print("  --------------------------------------")
            failure_count += 1
        except subprocess.TimeoutExpired:
            print(f"  [ERROR] Tool timed out on scene {scene_dir.name}.")
            failure_count += 1
        except Exception as e:
            print(f"  [CRITICAL ERROR] An unexpected error occurred while processing {scene_dir.name}: {e}")
            failure_count += 1
        
        print("-" * (len(scene_dir.name) + 30) + "\n")

    # --- Final Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    print("="*40)
    print("Batch Processing Complete")
    print("="*40)
    print(f"Total scenes processed: {success_count + failure_count}")
    print(f"  Successes: {success_count}")
    print(f"  Failures:  {failure_count}")
    print(f"Total time elapsed: {total_time:.2f} seconds")
    print("="*40)

if __name__ == '__main__':
    main()