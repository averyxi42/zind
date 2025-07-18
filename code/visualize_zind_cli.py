# """CLI script to visualize & validate data for the public-facing Zillow Indoor Dataset (ZInD).
#
# THIS VERSION HAS BEEN MODIFIED TO REMOVE THE OPENCV (cv2) DEPENDENCY.
# The floor plan visualization now uses Matplotlib exclusively.
#
# NEW FEATURE: Can now also visualize a pre-generated .npy grid map and save it as a JPEG.
# """

import argparse
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

# --- NEW IMPORTS for NPY and Matplotlib Rendering ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
# ---

# --- MODIFIED IMPORT: Removed the problematic render_jpg_image ---
from floor_plan import FloorPlan
from render import (
    render_room_vertices_on_panos,
    # render_jpg_image, # This function used OpenCV
    render_raster_to_vector_alignment,
)
from tqdm import tqdm

from utils import PolygonType, Polygon # Import for type hinting

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOG = logging.getLogger(__name__)

RENDER_FOLDER = "render_data"


# --- NEW MATPLOTLIB-BASED RENDER FUNCTION ---
def render_jpg_image_matplotlib(polygon_list: List[Polygon], jpg_file_name: str):
    """
    Renders a top-down view of a ZInD vector layout using Matplotlib,
    saving the output as a JPEG. This function has no OpenCV dependency.
    """
    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(Path(jpg_file_name).stem)

    # Define a clear color scheme for the elements
    colors = {
        'room': '#F0F0F0',
        'room_edge': '#222222',
        'door': '#966F33',
        'window': '#87CEEB',
        'opening': '#FFC0CB', # Pink for openings
        'primary_camera': '#FF0000', # Red
        'secondary_camera': '#FFA500' # Orange
    }
    
    all_points = []
    for poly in polygon_list:
        # Convert ZInD Point2D objects to a NumPy array for plotting
        points_np = np.array([[p.x, p.y] for p in poly.points])
        all_points.append(points_np)
        
        poly_type = poly.type
        if poly_type == PolygonType.ROOM:
            patch = patches.Polygon(points_np, closed=True, facecolor=colors['room'], edgecolor=colors['room_edge'], linewidth=1.5)
            ax.add_patch(patch)
        elif poly_type == PolygonType.DOOR:
            ax.plot(points_np[:, 0], points_np[:, 1], color=colors['door'], linewidth=4)
        elif poly_type == PolygonType.WINDOW:
            ax.plot(points_np[:, 0], points_np[:, 1], color=colors['window'], linewidth=6)
        elif poly_type == PolygonType.OPENING:
            ax.plot(points_np[:, 0], points_np[:, 1], color=colors['opening'], linewidth=4, linestyle='--')
        elif poly_type == PolygonType.PRIMARY_CAMERA:
            ax.plot(points_np[0, 0], points_np[0, 1], 'o', color=colors['primary_camera'], markersize=8)
        elif poly_type == PolygonType.SECONDARY_CAMERA:
            ax.plot(points_np[0, 0], points_np[0, 1], 'o', color=colors['secondary_camera'], markersize=6)

    # Automatically set plot limits
    if all_points:
        all_points_stacked = np.vstack(all_points)
        min_coords = np.min(all_points_stacked, axis=0)
        max_coords = np.max(all_points_stacked, axis=0)
        ax.set_xlim(min_coords[0] - 1, max_coords[0] + 1)
        ax.set_ylim(min_coords[1] - 1, max_coords[1] + 1)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(jpg_file_name, format='jpeg', quality=95)
    plt.close(fig)

# --- NEW NPY Visualization Function (already uses Matplotlib) ---
def render_npy_to_jpeg(npy_path: str, output_folder: str):
    """
    Loads a .npy grid map, applies a colormap, and saves it as a JPEG image.
    """
    npy_file = Path(npy_path)
    if not npy_file.is_file():
        LOG.error(f"NPY file not found at: {npy_path}")
        return

    LOG.info(f"Visualizing NPY file: {npy_file.name}")
    grid_map = np.load(npy_file)

    if grid_map.size == 0:
        LOG.warning("NPY file is empty. Skipping visualization.")
        return

    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    max_id = int(grid_map.max())
    cmap = plt.get_cmap('tab20b')
    num_colors = max_id + 1
    colors = cmap(np.arange(num_colors) % 20)
    colors[0] = [0.15, 0.15, 0.15, 1];
    if max_id >= 1: colors[1] = [0.9, 0.9, 0.9, 1];
    custom_cmap = ListedColormap(colors)
    ax.imshow(grid_map, cmap=custom_cmap, origin='lower', interpolation='none')
    ax.axis('off')
    output_filename = npy_file.stem + ".jpeg"
    output_path = Path(output_folder) / output_filename
    plt.tight_layout(pad=0)
    plt.savefig(output_path, format='jpeg', quality=95, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    LOG.info(f"Successfully saved visualization to: {output_path}")

def validate_and_render(
    zillow_floor_plan: "FloorPlan",
    *,
    input_folder: str,
    output_folder: str,
    args: Dict[str, Any]
):
    # (This function is mostly unchanged, but the call inside is modified)
    geometry_to_visualize = []
    if args.raw: geometry_to_visualize.append("raw")
    if args.complete: geometry_to_visualize.append("complete")
    if args.visible: geometry_to_visualize.append("visible")
    if args.redraw: geometry_to_visualize.append("redraw")

    panos_to_visualize = []
    if args.primary: panos_to_visualize.append("primary")
    if args.secondary: panos_to_visualize.append("secondary")

    if args.visualize_layout:
        for geometry_type in geometry_to_visualize:
            if geometry_type == "redraw": continue
            for pano_type in panos_to_visualize:
                output_folder_layout = os.path.join(output_folder, "layout", geometry_type, pano_type)
                os.makedirs(output_folder_layout, exist_ok=True)
                panos_list = zillow_floor_plan.panos_layouts[geometry_type][pano_type]
                render_room_vertices_on_panos(
                    input_folder=zillow_floor_plan.input_folder,
                    panos_list=panos_list,
                    output_folder=output_folder_layout,
                )

    if args.visualize_floor_plan:
        output_folder_floor_plan = os.path.join(output_folder, "floor_plan")
        os.makedirs(output_folder_floor_plan, exist_ok=True)
        for geometry_type in geometry_to_visualize:
            if geometry_type == "visible": continue
            zind_dict = zillow_floor_plan.floor_plan_layouts[geometry_type]
            for floor_id, zind_poly_list in zind_dict.items():
                output_file_name = os.path.join(output_folder_floor_plan, f"vector_{geometry_type}_layout_{floor_id}.jpg")
                # --- MODIFIED CALL: Use our new matplotlib function ---
                render_jpg_image_matplotlib(polygon_list=zind_poly_list, jpg_file_name=output_file_name)

    if args.visualize_raster:
        output_folder_floor_plan_alignment = os.path.join(output_folder, "floor_plan_raster_to_vector_alignment")
        os.makedirs(output_folder_floor_plan_alignment, exist_ok=True)
        for geometry_type in geometry_to_visualize:
            if geometry_type == "visible": continue
            for floor_id, raster_to_vector_transformation in zillow_floor_plan.floor_plan_to_redraw_transformation.items():
                floor_plan_image_path = os.path.join(input_folder, zillow_floor_plan.floor_plan_image_path[floor_id])
                zind_poly_list = zillow_floor_plan.floor_plan_layouts[geometry_type][floor_id]
                output_file_name = os.path.join(output_folder_floor_plan_alignment, f"raster_to_vector_{geometry_type}_layout_{floor_id}.jpg")
                render_raster_to_vector_alignment(zind_poly_list, raster_to_vector_transformation, floor_plan_image_path, output_file_name)


def main():
    # (The main function is unchanged from your last version)
    parser = argparse.ArgumentParser(description="Visualize & validate Zillow Indoor Dataset (ZInD)")
    zind_group = parser.add_argument_group('ZInD JSON Processing')
    zind_group.add_argument("--input", "-i", help="Input JSON file (or folder with ZInD data)")
    zind_group.add_argument("--visualize-layout", action="store_true", help="Render room vertices and WDO on panoramas.")
    zind_group.add_argument("--visualize-floor-plan", action="store_true", help="Render the floor plans as top-down projections.")
    zind_group.add_argument("--visualize-raster", action="store_true", help="Render the vector floor plan on the raster floor plan image.")
    zind_group.add_argument("--max-tours", type=float, default=float("inf"), help="Max tours to process.")
    zind_group.add_argument("--primary", action="store_true", help="Visualize primary panoramas.")
    zind_group.add_argument("--secondary", action="store_true", help="Visualize secondary panoramas.")
    zind_group.add_argument("--raw", action="store_true", help="Visualize raw layout.")
    zind_group.add_argument("--complete", action="store_true", help="Visualize complete layout.")
    zind_group.add_argument("--visible", action="store_true", help="Visualize visible layout.")
    zind_group.add_argument("--redraw", action="store_true", help="Visualize 2D redraw geometry.")
    npy_group = parser.add_argument_group('NPY Grid Map Visualization')
    npy_group.add_argument("--visualize_npy", type=str, help="Path to a .npy grid map file to visualize as a JPEG.")
    parser.add_argument("--output", "-o", help="Output folder where rendered data will be saved to", required=True)
    parser.add_argument("--debug", "-d", action="store_true", help="Set log level to DEBUG")
    args = parser.parse_args()

    if args.debug:
        LOG.setLevel(logging.DEBUG)
    os.makedirs(args.output, exist_ok=True)

    if args.visualize_npy:
        render_npy_to_jpeg(args.visualize_npy, args.output)
        return
    if not args.input:
        parser.error("--input is required unless --visualize_npy is used.")

    input_path = args.input
    max_tours_to_process = args.max_tours
    input_files_list = [input_path]
    if Path(input_path).is_dir():
        input_files_list = sorted(Path(input_path).glob("**/zind_data.json"))

    num_failed, num_success, failed_tours = 0, 0, []
    for input_file in tqdm(input_files_list, desc="Validating ZInD data"):
        try:
            zillow_floor_plan = FloorPlan(input_file)
            current_input_folder = str(Path(input_file).parent)
            current_output_folder = os.path.join(args.output, RENDER_FOLDER, Path(input_file).parent.stem)
            os.makedirs(current_output_folder, exist_ok=True)
            validate_and_render(zillow_floor_plan, input_folder=current_input_folder, output_folder=current_output_folder, args=args)
            num_success += 1
            if num_success >= max_tours_to_process:
                LOG.info(f"Max tours to process reached {num_success}")
                break
        except Exception as ex:
            failed_tours.append(str(Path(input_file).parent.stem))
            num_failed += 1
            track = traceback.format_exc()
            LOG.warning(f"Error validating {input_file}: {ex}")
            LOG.debug(track)
            continue
    if num_failed > 0:
        LOG.warning(f"Failed to validate: {num_failed}")
        LOG.debug(f"Failed_tours: {failed_tours}")
    else:
        LOG.info("All ZInD validated successfully")

if __name__ == "__main__":
    main()