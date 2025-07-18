"""
ZInD Batch Processing Tool (Parallelized)

This script finds and processes all Zillow Indoor Dataset JSON files in parallel,
calling the main generation tool for each one. It uses a process pool to fully
utilize multi-core CPUs, significantly accelerating the workflow.

Features:
- **Parallel Execution:** Uses a `ProcessPoolExecutor` to run multiple processing
  jobs simultaneously.
- **Progress Bar:** Provides a real-time `tqdm` progress bar to monitor the
  status of the entire batch job.
- **Configurable Workers:** The number of parallel workers can be configured,
  defaulting to the number of CPU cores on the machine.
- **Robust Error Handling:** Failures in one scene will be logged without
  crashing the entire batch process.
"""
import sys
import argparse
import subprocess
from pathlib import Path
import time
import os
import concurrent.futures
from tqdm import tqdm

def process_scene(scene_dir: Path, tool_script: Path, args: argparse.Namespace):
    """
    A single worker function that processes one scene.
    This function is designed to be called by a ProcessPoolExecutor.
    """
    try:
        input_json_path = scene_dir / args.filename
        if not input_json_path.is_file():
            return "skip_json_missing", scene_dir, f"JSON file '{args.filename}' not found."

        output_base_name = f"map_res{args.resolution}_{args.subdivision_method}"
        if args.merge_unlabeled:
            output_base_name += "_merged"
        output_npy_path = scene_dir / f"{output_base_name}.npy"
        
        # Check for existing files if not overwriting
        if output_npy_path.exists() and not args.force_overwrite:
            return "skip_exists", scene_dir, f"Output file already exists: {output_npy_path.name}"

        command = [
            sys.executable,
            str(tool_script),
            str(input_json_path),
            "--output_path", str(output_npy_path),
            "--resolution", str(args.resolution),
            "--subdivision_method", args.subdivision_method,
            "--no_viz"
        ]
        if args.merge_unlabeled: command.append("--merge_unlabeled")
        if args.viz_vector: command.append("--viz_vector")
        if args.save_visualizations: command.append("--save_visualizations")

        result = subprocess.run(
            command,
            check=True,
            timeout=300
        )
        return "success", scene_dir, result.stdout # Return stdout for potential verbose logging
    
    except subprocess.CalledProcessError as e:
        error_message = f"Tool failed with exit code {e.returncode}.\nSTDERR:\n{e.stderr}"
        return "failure", scene_dir, error_message
    except subprocess.TimeoutExpired:
        return "failure", scene_dir, "Process timed out after 300 seconds."
    except Exception as e:
        return "failure", scene_dir, f"An unexpected critical error occurred: {e}"

def main():
    parser = argparse.ArgumentParser(
        description="ZInD Batch Processing Tool (Parallelized).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- Core arguments ---
    parser.add_argument("root_directory", type=str, help="The root directory containing scene subdirectories.")
    parser.add_argument("tool_script_path", type=str, help="Path to the main grid map generation script.")
    parser.add_argument("--filename", type=str, default="zind_data.json", help="The name of the JSON file to find.")
    
    # --- Parallelism Control ---
    parser.add_argument("--num_workers", type=int, default=os.cpu_count(),
                        help=f"Number of parallel processes to use (default: all available cores, {os.cpu_count()}).")

    # --- Passthrough arguments for the main tool ---
    parser.add_argument("-r", "--resolution", type=float, default=0.04, help="Grid resolution to pass to the tool.")
    parser.add_argument("--subdivision_method", type=str, choices=['none', 'raw', 'voronoi'], default='raw', help="Subdivision algorithm to pass to the tool.")
    parser.add_argument("--merge_unlabeled", action="store_true", help="Pass '--merge_unlabeled' to the tool.")
    parser.add_argument("--viz_vector", action="store_true", help="Pass '--viz_vector' to the tool.")
    parser.add_argument("--save_visualizations", action="store_true", help="Pass '--save_visualizations' to the tool.")

    # --- Workflow Control ---
    parser.add_argument("--force_overwrite", action="store_true", help="Force reprocessing and overwrite existing output files.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands that would be executed without running them.")
    args = parser.parse_args()

    root_dir = Path(args.root_directory); tool_script = Path(args.tool_script_path)
    if not root_dir.is_dir(): print(f"Error: Root directory not found: '{root_dir.resolve()}'"); sys.exit(1)
    if not tool_script.is_file(): print(f"Error: Tool script not found: '{tool_script.resolve()}'"); sys.exit(1)

    scene_dirs = sorted([d for d in root_dir.glob('*') if d.is_dir()])
    total_scenes = len(scene_dirs)
    print(f"Found {total_scenes} potential scene directories in '{root_dir}'.")
    print(f"Using up to {args.num_workers} parallel workers.\n")

    # Dry run is simple and sequential, no need for parallel overhead
    if args.dry_run:
        print("--- DRY RUN MODE ---")
        for scene_dir in scene_dirs:
            input_json_path = scene_dir / args.filename
            if not input_json_path.is_file(): continue
            output_base_name = f"map_res{args.resolution}_{args.subdivision_method}{'_merged' if args.merge_unlabeled else ''}"
            output_npy_path = scene_dir / f"{output_base_name}.npy"
            command = [sys.executable, str(tool_script), str(input_json_path), "--output_path", str(output_npy_path), "--resolution", str(args.resolution), "--subdivision_method", args.subdivision_method, "--no_viz"]
            if args.merge_unlabeled: command.append("--merge_unlabeled")
            if args.viz_vector: command.append("--viz_vector")
            if args.save_visualizations: command.append("--save_visualizations")
            print(f"  Would process {scene_dir.name}: {' '.join(command)}")
        return

    # --- Main Parallel Processing Loop ---
    success_count, failure_count, skip_count = 0, 0, 0
    failed_scenes = []
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # Submit all jobs to the pool
        futures = {executor.submit(process_scene, scene_dir, tool_script, args): scene_dir for scene_dir in scene_dirs}
        
        # Use tqdm for a live progress bar as futures complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_scenes, desc="Processing Scenes"):
            status, scene_dir, message = future.result()
            
            if status == "success":
                success_count += 1
            elif status.startswith("skip"):
                skip_count += 1
            else: # failure
                failure_count += 1
                failed_scenes.append((scene_dir.name, message))

    # --- Final Summary ---
    end_time = time.time()
    total_time = end_time - start_time
    print("\n" + "="*40)
    print("Batch Processing Complete")
    print("="*40)
    print(f"Total scenes found: {total_scenes}")
    print(f"  Processed successfully: {success_count}")
    print(f"  Skipped (exist/no json):{skip_count}")
    print(f"  Failures:               {failure_count}")
    print(f"Total time elapsed: {total_time:.2f} seconds")
    print("="*40)
    
    if failed_scenes:
        print("\n--- FAILED SCENES ---")
        for name, msg in failed_scenes:
            print(f"\nScene: {name}")
            print(f"Error: {msg}")
        print("\n" + "="*40)

if __name__ == '__main__':
    main()