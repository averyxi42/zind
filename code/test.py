"""
ZInD Floor Plan Visualization Tool

This script loads a Zillow Indoor Dataset (ZInD) floor plan from its primary
JSON file and generates a 2D visualization of the final 'redraw' layout using
Matplotlib.

Dependencies:
1.  Python 3.6+
2.  Matplotlib (`pip install matplotlib`)
3.  NumPy (`pip install numpy`)
4.  The ZInD utility scripts (`floor_plan.py`, `utils.py`, `transformations.py`)
    must be accessible in your Python environment (e.g., in the same directory
    or in your PYTHONPATH).

Usage:
    python draw_zind_floorplan.py /path/to/your/scene.json
"""

import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- ZInD Utility Imports ---
# This is the first and most critical check. The script will not proceed
# without these helper files from the official ZInD repository.
sys.path.append('.')
from floor_plan import FloorPlan
from utils import PolygonType



def draw_floorplan_from_json(json_filepath: str):
    """
    Loads a ZInD floor plan and draws the 'redraw' layout using Matplotlib.

    Args:
        json_filepath (str): The full path to the ZInD JSON file for a scene.
    
    Raises:
        Exception: Can raise various exceptions if the JSON is malformed or
                   does not contain the expected structure.
    """
    print(f"Loading floor plan from: {json_filepath}")
    
    # 1. Instantiate the FloorPlan object from the real ZInD JSON file.
    fp = FloorPlan(json_filepath)

    # 2. Select the final, clean geometry ('redraw').
    redraw_layouts = fp.floor_plan_layouts.get("redraw")

    if not redraw_layouts:
        print("Error: No 'redraw' layouts found in the provided JSON file.")
        return

    # Select the first available floor (typically '0' or '1').
    floor_id = list(redraw_layouts.keys())[0]
    polygons_to_draw = redraw_layouts[floor_id]
    
    if not polygons_to_draw:
        print(f"Warning: Floor '{floor_id}' found, but it contains no polygons to draw.")
        return
        
    print(f"Drawing floor '{floor_id}' with {len(polygons_to_draw)} geometric elements...")
    
    # 3. Set up the Matplotlib plot.
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(f"ZInD Floor Plan: {Path(json_filepath).stem}")
    ax.set_aspect('equal', adjustable='box')

    # Define a clear color scheme for the elements.
    colors = {
        'room': '#F0F0F0',       # A very light gray for room fill
        'room_edge': '#222222',  # Dark gray for walls
        'door': '#966F33',       # A wooden brown for doors
        'window': '#87CEEB'      # Sky blue for windows
    }

    # 4. Iterate through the real polygons and draw them based on their type.
    for poly in polygons_to_draw:
        poly_type = poly.type
        points = poly.points

        if poly_type == PolygonType.ROOM:
            room_patch = patches.Polygon(
                points,
                closed=True,
                facecolor=colors['room'],
                edgecolor=colors['room_edge'],
                linewidth=1.5,
                label='Room' # For potential legend
            )
            ax.add_patch(room_patch)

        elif poly_type == PolygonType.DOOR:
            p1, p2 = points
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors['door'], linewidth=4)

        elif poly_type == PolygonType.WINDOW:
            p1, p2 = points
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=colors['window'], linewidth=6)
            
        elif poly_type == PolygonType.PIN_LABEL:
            position = points[0]
            label_text = poly.name
            ax.text(position[0], position[1], label_text, 
                    ha='center', va='center', fontsize=9, color='black',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

    ax.grid(True, which='both', linestyle=':', linewidth=0.6)
    print("Displaying plot...")
    plt.show()


def main():
    """
    Main function to parse command-line arguments and run the visualization.
    """
    parser = argparse.ArgumentParser(
        description="ZInD Floor Plan Visualization Tool.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example:\n  python draw_zind_floorplan.py /path/to/zind/data/some_scene.json"
    )
    
    parser.add_argument(
        "json_filepath",
        type=str,
        help="The full path to the ZInD JSON file for a single scene."
    )
    args = parser.parse_args()
    
    # --- Input Validation ---
    filepath = Path(args.json_filepath)
    if not filepath.is_file():
        print(f"\nError: The provided path does not exist or is not a file.")
        print(f"Checked path: {filepath.resolve()}\n")
        sys.exit(1)
        
    # --- Execute Drawing ---
    try:
        draw_floorplan_from_json(args.json_filepath)
    except Exception as e:
        print(f"\nAn unexpected error occurred while processing the file: {e}")
        print("Please ensure the file is a valid, uncorrupted ZInD JSON file.\n")
        sys.exit(1)


if __name__ == '__main__':
    main()