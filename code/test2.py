"""
ZInD Color-Coded Raw Floor Plan Visualization Tool

This script loads a Zillow Indoor Dataset (ZInD) floor plan from its primary
JSON file. It visualizes the 'raw' layout geometry, color-coding each room
based on its semantic type (e.g., "Kitchen", "Bedroom").

The room types are derived from the 'redraw' layout's pin labels, which act
as the semantic ground truth.

Dependencies:
1.  Python 3.6+
2.  Matplotlib (`pip install matplotlib`)
3.  NumPy (`pip install numpy`)
4.  SciPy (`pip install scipy`) for spatial lookups.
5.  The ZInD utility scripts (`floor_plan.py`, `utils.py`, etc.) must be
    accessible in your Python environment.

Usage:
    python draw_zind_raw_color.py /path/to/your/scene.json
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import cKDTree

# --- ZInD Utility Imports ---

from floor_plan import FloorPlan
from utils import PolygonType



def get_room_label_map(fp: FloorPlan, floor_id: str) -> dict:
    """
    Builds a map of room labels from the 'redraw' layout's pin labels.

    This uses a k-d tree for efficient spatial lookups.

    Returns:
        A tuple containing: (k-d tree of pin locations, list of pin labels)
    """
    print("...Building room label map from 'redraw' data...")
    redraw_layout = fp.floor_plan_layouts.get("redraw", {}).get(floor_id, [])
    
    pin_locations = []
    pin_labels = []
    for poly in redraw_layout:
        if poly.type == PolygonType.PIN_LABEL:
            pin_locations.append(poly.points[0])
            pin_labels.append(poly.name)

    if not pin_labels:
        print("Warning: No pin labels found in 'redraw' layout. Rooms will not be colored.")
        return None, None
        
    print(f"...Found {len(pin_labels)} room labels: {set(pin_labels)}")
    return cKDTree(pin_locations), pin_labels


def get_room_color(room_type: str, color_map: dict) -> str:
    """
    Assigns a consistent color to each room type.
    """
    if room_type not in color_map:
        # Predefined list of visually distinct colors
        predefined_colors = [
            '#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
            '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928',
            '#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462'
        ]
        # Assign a new color from the list or cycle through if we run out
        color_map[room_type] = predefined_colors[len(color_map) % len(predefined_colors)]
    return color_map[room_type]

def draw_raw_floorplan_color_coded(json_filepath: str):
    """
    Loads and draws the 'raw' layout, with rooms colored by type.
    """
    print(f"Loading floor plan from: {json_filepath}")
    fp = FloorPlan(json_filepath)

    # We will use the 'raw' layout for geometry
    raw_layouts = fp.floor_plan_layouts.get("redraw")
    if not raw_layouts:
        print("Error: No 'raw' layouts found in the provided JSON file.")
        return

    floor_id = list(raw_layouts.keys())[1]
    raw_polygons = raw_layouts[floor_id]
    
    # 1. Build the semantic map from the 'redraw' layout
    label_kdtree, labels = get_room_label_map(fp, floor_id)
    
    print(f"Drawing 'raw' layout for floor '{floor_id}' with {len(raw_polygons)} elements...")

    # 2. Set up plot
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(f"ZInD Raw Floor Plan (Color Coded): {Path(json_filepath).stem}")
    ax.set_aspect('equal', adjustable='box')
    
    room_color_map = {}
    legend_handles = []
    plotted_labels = set()

    # 3. Iterate through RAW polygons and draw them
    for poly in raw_polygons:
        if poly.type == PolygonType.ROOM:
            room_centroid = np.mean(poly.points, axis=0)
            room_type = "Unknown"
            
            if label_kdtree:
                # Find the closest pin label to this room's centroid
                distance, index = label_kdtree.query(room_centroid)
                room_type = labels[index]

            color = get_room_color(room_type, room_color_map)
            
            # Draw the room polygon
            room_patch = patches.Polygon(
                poly.points, closed=True, facecolor=color, edgecolor='black', linewidth=1.0
            )
            ax.add_patch(room_patch)

            # Add a label inside the room
            ax.text(room_centroid[0], room_centroid[1], room_type, ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

            # Create legend handles for each unique room type
            if room_type not in plotted_labels:
                legend_handles.append(patches.Patch(color=color, label=room_type))
                plotted_labels.add(room_type)

    if legend_handles:
        ax.legend(handles=legend_handles, title="Room Types", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    else:
        plt.tight_layout()

    ax.grid(True, linestyle=':', alpha=0.6)
    print("Displaying plot...")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="ZInD Color-Coded Raw Floor Plan Visualization Tool.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example:\n  python draw_zind_raw_color.py /path/to/zind/data/some_scene.json"
    )
    parser.add_argument(
        "json_filepath", type=str, help="The full path to the ZInD JSON file for a single scene."
    )
    args = parser.parse_args()
    
    filepath = Path(args.json_filepath)
    if not filepath.is_file():
        print(f"\nError: File not found at '{filepath.resolve()}'\n")
        sys.exit(1)
        
    try:
        draw_raw_floorplan_color_coded(args.json_filepath)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()