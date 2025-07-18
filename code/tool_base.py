"""
ZInD Robustly Connected Grid Map Generation Tool (v3)

Implements the user-provided superior method for creating traversable doorways
by rasterizing the convex hull of paired door polygons. This ensures robust,
geometrically accurate connectivity.

ID Scheme:
- 0: Obstacle (walls).
- 1: Traversable Space (doorways).
- 2, 3, 4...: Unique Room IDs.
"""

import sys
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from skimage.draw import polygon as draw_polygon
from scipy.spatial import ConvexHull # Import for the new method

# --- ZInD Utility Imports ---
try:
    from floor_plan import FloorPlan
    from utils import PolygonType, Point2D
except ImportError:
    print("\nFATAL ERROR: Could not import required ZInD utility classes.")
    print("Please ensure the ZInD utility scripts are in your PYTHONPATH.\n")
    sys.exit(1)


def find_and_rasterize_door_connections(grid_map, door_polygons, min_coords, resolution):
    """
    Finds pairs of facing doors and rasterizes the convex hull of their
    combined points to create a robust, traversable doorway.
    """
    print("5a. Finding door pairs to establish connectivity...")
    TRAVERSABLE_ID = 1
    DOOR_DISTANCE_THRESHOLD = 0.3
    DOOR_ANGLE_TOLERANCE = math.radians(10)

    num_doors = len(door_polygons)
    paired_indices = set()
    num_pairs_found = 0

    for i in range(num_doors):
        if i in paired_indices:
            continue

        door1 = door_polygons[i]
        points1_np = np.array([[p.x, p.y] for p in door1.points])
        center1 = np.mean(points1_np, axis=0)
        vec1 = points1_np[1] - points1_np[0]

        for j in range(i + 1, num_doors):
            if j in paired_indices:
                continue

            door2 = door_polygons[j]
            points2_np = np.array([[p.x, p.y] for p in door2.points])
            center2 = np.mean(points2_np, axis=0)
            vec2 = points2_np[1] - points2_np[0]
            
            dist = np.linalg.norm(center1 - center2)
            if dist > DOOR_DISTANCE_THRESHOLD:
                continue

            norm_vec1 = vec1 / np.linalg.norm(vec1)
            norm_vec2 = vec2 / np.linalg.norm(vec2)
            dot_product = abs(np.dot(norm_vec1, norm_vec2))
            if 1.0 - dot_product > math.sin(DOOR_ANGLE_TOLERANCE):
                continue
            
            # --- SUPERIOR METHOD IMPLEMENTATION ---
            # A pair has been found. Now, use the convex hull.
            num_pairs_found += 1
            paired_indices.add(i)
            paired_indices.add(j)

            # 1. Combine all 4 points from both door polygons.
            all_door_points = np.vstack([points1_np, points2_np])
            
            # 2. Compute the convex hull of these 4 points.
            hull = ConvexHull(all_door_points)
            
            # 3. Get the vertices of the hull in world coordinates.
            hull_vertices_world = all_door_points[hull.vertices]
            
            # 4. Convert hull vertices to grid coordinates.
            hull_vertices_grid = (hull_vertices_world - min_coords) / resolution
            
            # 5. Rasterize the resulting hull polygon.
            rr, cc = draw_polygon(hull_vertices_grid[:, 1], hull_vertices_grid[:, 0], shape=grid_map.shape)
            grid_map[rr, cc] = TRAVERSABLE_ID
            
            break # Move to the next unpaired door

    print(f"5b. Found and connected {num_pairs_found} internal door pairs using convex hull method.")
    num_isolated = num_doors - len(paired_indices)
    print(f"5c. Ignored {num_isolated} isolated (external) doors.")


def create_robust_grid_map(json_filepath: str, resolution: float = 0.05):
    """
    Generates a 2D grid map with unique room IDs and robustly connected doors.
    """
    print("1. Loading and parsing ZInD JSON file...")
    fp = FloorPlan(json_filepath)
    
    redraw_layouts = fp.floor_plan_layouts.get("redraw")
    if not redraw_layouts: raise ValueError("No 'redraw' layouts found.")
    
    floor_id = list(redraw_layouts.keys())[0]
    all_polygons = redraw_layouts[floor_id]
    
    room_polygons = [p for p in all_polygons if p.type == PolygonType.ROOM]
    door_polygons = [p for p in all_polygons if p.type == PolygonType.DOOR]
    
    print(f"2. Found {len(room_polygons)} rooms and {len(door_polygons)} doors. Determining bounds...")

    all_room_points_np = [np.array([[p.x, p.y] for p in poly.points]) for poly in room_polygons]
    all_points_stacked = np.vstack(all_room_points_np)
    min_coords = np.min(all_points_stacked, axis=0) - (2 * resolution)
    max_coords = np.max(all_points_stacked, axis=0) + (2 * resolution)

    grid_dims = (np.ceil((max_coords - min_coords) / resolution)).astype(int)
    grid_height, grid_width = grid_dims[1], grid_dims[0]
    
    print(f"3. Creating grid of size {grid_width}x{grid_height} (WxH pixels)...")
    grid_map = np.zeros((grid_height, grid_width), dtype=np.uint16)

    print("4. Rasterizing rooms...")
    for i, room_poly_np in enumerate(all_room_points_np):
        room_id = i + 2
        poly_grid = (room_poly_np - min_coords) / resolution
        rr, cc = draw_polygon(poly_grid[:, 1], poly_grid[:, 0], shape=grid_map.shape)
        grid_map[rr, cc] = room_id
    
    # 5. Find door pairs and draw robust connections between them using the convex hull method.
    find_and_rasterize_door_connections(grid_map, door_polygons, min_coords, resolution)

    print("6. Grid map generation complete.")
    return grid_map, {} # Metadata omitted for brevity


def main():
    parser = argparse.ArgumentParser(description="ZInD Robustly Connected Grid Map Generation Tool (v3).")
    parser.add_argument("json_filepath", type=str, help="Path to the ZInD JSON file.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the grid map as a .npy file.")
    parser.add_argument("-r", "--resolution", type=float, default=0.02, help="Grid resolution in meters/pixel.")
    parser.add_argument("--no_viz", action="store_true", help="Suppress the matplotlib visualization.")
    args = parser.parse_args()

    if not Path(args.json_filepath).is_file():
        print(f"\nError: File not found at '{Path(args.json_filepath).resolve()}'\n")
        sys.exit(1)
    
    try:
        grid_map, _ = create_robust_grid_map(args.json_filepath, args.resolution)
        
        if args.output_path:
            np.save(args.output_path, grid_map)
            print(f"Successfully saved grid map to: {args.output_path}")

        if not args.no_viz:
            visualize_grid_map(grid_map, {}, Path(args.json_filepath).name)
            
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def visualize_grid_map(grid_map: np.ndarray, metadata: dict, json_filename: str):
    # Visualization function remains the same as before
    if grid_map.size == 0: return
    fig, ax = plt.subplots(figsize=(12, 12))
    max_id = int(grid_map.max())
    if max_id < 2: max_id = 2
    base_cmap = plt.get_cmap('tab20', max_id + 1)
    colors = base_cmap(np.linspace(0, 1, max_id + 1))
    colors[0] = np.array([0.1, 0.1, 0.1, 1])
    colors[1] = np.array([0.9, 0.9, 0.9, 1]) # Light gray for traversable to distinguish from pure white
    custom_cmap = ListedColormap(colors)
    ax.imshow(grid_map, cmap=custom_cmap, origin='lower', interpolation='none')
    ax.set_title(f"Robustly Connected Grid Map (Convex Hull) for: {json_filename}")
    plt.show()

if __name__ == '__main__':
    main()