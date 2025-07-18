"""
ZInD Advanced Grid Map Generation Tool (v29 - Hole Filling Fix)

This definitive version corrects a critical bug where the union of overlapping
raw polygons would create unintentional "holes" that were then incorrectly
rasterized as obstacles.

Key Correction:
- **Hole Filling Logic:** A new `_fill_holes` helper function is introduced.
  After any `unary_union` operation that produces a final room shape, this
  function is called to reconstruct the polygon using only its exterior
  boundary. This geometrically "fills" any holes created by fully-contained
  raw polygons, ensuring the entire room area is correctly labeled.
- **Complete & Self-Contained:** A complete, runnable script with all features.
"""

import sys
import argparse
import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

# --- Dependency Checks ---
try:
    from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, MultiPoint, MultiPolygon, LineString as ShapelyLineString
    from shapely.ops import unary_union, voronoi_diagram
except ImportError:
    print("\nFATAL ERROR: This script requires 'shapely'. Install with: pip install shapely\n"); sys.exit(1)
try:
    from skimage.draw import polygon as draw_polygon
    from scipy.spatial import ConvexHull
except ImportError:
    print("\nFATAL ERROR: This script requires 'scikit-image' and 'scipy'. Install with: pip install scikit-image scipy\n"); sys.exit(1)
try:
    from floor_plan import FloorPlan
    from utils import PolygonType, Point2D, Polygon
except ImportError:
    print("\nFATAL ERROR: Could not import ZInD utility classes. Ensure they are in your PYTHONPATH.\n"); sys.exit(1)

# --- Constants ---
RAW_FALLBACK_IOU_THRESHOLD = 0.95

def to_shapely(poly: Polygon):
    return ShapelyPolygon([(p.x, p.y) for p in poly.points])

def _fill_holes(shape):
    """
    Takes a Shapely geometry and reconstructs any polygons within it using
    only their exterior boundaries, effectively filling any interior holes.
    """
    if isinstance(shape, ShapelyPolygon) and list(shape.interiors):
        return ShapelyPolygon(list(shape.exterior.coords))
    if isinstance(shape, MultiPolygon):
        return unary_union([ShapelyPolygon(list(p.exterior.coords)) for p in shape.geoms])
    return shape

def _get_adjacent_regions(piece, candidates):
    neighbors = []
    for i, region in enumerate(candidates):
        if piece.buffer(1e-9).intersects(region.buffer(1e-9)):
            try:
                boundary_len = piece.intersection(region).length
                if boundary_len > 1e-6: neighbors.append({'index': i, 'boundary': boundary_len})
            except Exception: continue
    return sorted(neighbors, key=lambda x: x['boundary'], reverse=True)

def _find_and_pair_doorways(doors: list) -> list:
    """
    Identifies pairs of internal doors and returns their connecting polygons.
    This definitive version uses a try-except block to safely use ConvexHull,
    falling back to a robust method for collinear/degenerate points.
    """
    print("......finding door pairs...")
    doorway_polys, paired_indices = [], set()
    for i in range(len(doors)):
        if i in paired_indices: continue
        p1_np = np.array([[p.x, p.y] for p in doors[i].points])
        for j in range(i + 1, len(doors)):
            if j in paired_indices: continue
            p2_np = np.array([[p.x, p.y] for p in doors[j].points])
            
            # Check for proximity and alignment
            if np.linalg.norm(np.mean(p1_np, axis=0) - np.mean(p2_np, axis=0)) > 0.3: continue
            v1,v2=p1_np[1]-p1_np[0],p2_np[1]-p2_np[0]
            norm_v1 = v1 / np.linalg.norm(v1); norm_v2 = v2 / np.linalg.norm(v2)
            if 1.0 - abs(np.dot(norm_v1, norm_v2)) > math.sin(math.radians(10)): continue
            
            paired_indices.add(i); paired_indices.add(j)
            
            all_points = np.vstack([p1_np, p2_np])
            
            try:
                # --- Attempt the primary, preferred method ---
                hull = ConvexHull(all_points)
                # Ensure the hull is not just a line (has area)
                if hull.volume > 1e-9: # 'volume' is area for 2D points
                    doorway_poly = ShapelyPolygon(all_points[hull.vertices])
                else:
                    # Raise an error to trigger the fallback for flat hulls
                    raise Exception("Hull is flat, fallback needed.")
            except:
                # --- Fallback for collinear or degenerate points ---
                print(f"      [Info] ConvexHull failed for door pair (likely collinear). Falling back to robust method.")
                # Manual construction is safe and handles this case perfectly.
                doorway_poly = ShapelyPolygon([
                    (p1_np[0][0], p1_np[0][1]),
                    (p1_np[1][0], p1_np[1][1]),
                    (p2_np[1][0], p2_np[1][1]),
                    (p2_np[0][0], p2_np[0][1])
                ])

            doorway_polys.append(doorway_poly)
            break
            
    return doorway_polys
def _subdivide_by_voronoi(master_shape, room_pins):
    regions = []
    pin_points = [ShapelyPoint(p.points[0].x, p.points[0].y) for p in room_pins]
    if not pin_points: return [[master_shape, "Unlabeled_Area"]]
    try: voronoi_cells = voronoi_diagram(MultiPoint(pin_points), envelope=master_shape)
    except Exception as e:
        print(f"      [Warning] Voronoi diagram failed: {e}. Not subdividing."); return [[master_shape, " / ".join(sorted([p.name for p in room_pins]))]]

    for cell in voronoi_cells.geoms:
        final_cell = master_shape.intersection(cell)
        if final_cell.is_empty: continue
        cell_center = final_cell.representative_point()
        closest_pin_idx = min(range(len(pin_points)), key=lambda k: cell_center.distance(pin_points[k]))
        pin_name = room_pins[closest_pin_idx].name
        decomposed_polys = list(final_cell.geoms) if isinstance(final_cell, MultiPolygon) else [final_cell]
        for poly in decomposed_polys:
            if not poly.is_empty: regions.append([_fill_holes(poly), pin_name])
    return regions

def _subdivide_by_raw(master_shape, room_pins, raw_rooms):
    successful_sub_regions, successful_sub_names, failed_pins = [], [], []
    for pin in room_pins:
        pin_point = ShapelyPoint(pin.points[0].x, pin.points[0].y)
        other_important_pins = [p for p in room_pins if p != pin and not (p.name.startswith("Room_") and p.name.split('_')[-1].isdigit())]
        guiding_raw_polys = [s for s in raw_rooms if s.buffer(1e-9).contains(pin_point)]
        is_ambiguous = any(raw_poly.buffer(1e-9).contains(ShapelyPoint(p.points[0].x, p.points[0].y)) for raw_poly in guiding_raw_polys for p in other_important_pins)
        
        if not is_ambiguous and guiding_raw_polys:
            sub_region = unary_union(guiding_raw_polys).intersection(master_shape)
            # --- CRITICAL FIX: Fill holes created by the union ---
            sub_region = _fill_holes(sub_region)
            decomposed_polys = list(sub_region.geoms) if isinstance(sub_region, MultiPolygon) else [sub_region]
            for poly in decomposed_polys:
                if not poly.is_empty: successful_sub_regions.append(poly); successful_sub_names.append(pin.name)
        else:
            if is_ambiguous: print(f"      [Info] Raw guide for pin '{pin.name}' is ambiguous. Queuing for Voronoi.")
            failed_pins.append(pin)
    
    if failed_pins:
        remaining_area = master_shape.difference(unary_union(successful_sub_regions))
        if not remaining_area.is_empty:
            voronoi_sub_regions = _subdivide_by_voronoi(remaining_area, failed_pins)
            for shape, name in voronoi_sub_regions:
                successful_sub_regions.append(shape); successful_sub_names.append(name)
    
    leftover_area = master_shape.difference(unary_union(successful_sub_regions))
    return successful_sub_regions, successful_sub_names, leftover_area

def _subdivide_and_label_rooms(redraw_rooms, redraw_pins, raw_rooms, merge_slivers, subdivision_method: str):
    print(f"......subdivision method: {subdivision_method.upper()}")
    pins_in_room, base_regions, unmerged_slivers = {}, [], []
    for pin in redraw_pins:
        pin_point = ShapelyPoint(pin.points[0].x, pin.points[0].y)
        for i, room_shape in enumerate(redraw_rooms):
            if room_shape.buffer(1e-9).contains(pin_point): pins_in_room.setdefault(i, []).append(pin); break
    
    for i, master_shape in enumerate(redraw_rooms):
        room_pins = pins_in_room.get(i, [])
        effective_method = subdivision_method
        if subdivision_method == 'raw' and len(room_pins) > 1:
            all_guiding_raw_polys = [s for pin in room_pins for s in raw_rooms if s.buffer(1e-9).contains(ShapelyPoint(pin.points[0].x, pin.points[0].y))]
            if all_guiding_raw_polys:
                raw_union_shape = unary_union(all_guiding_raw_polys).intersection(master_shape)
                iou = raw_union_shape.area / master_shape.area if master_shape.area > 1e-9 else 0
                if iou > RAW_FALLBACK_IOU_THRESHOLD:
                    print(f"      [Info] Raw geometry IoU > {RAW_FALLBACK_IOU_THRESHOLD}. Falling back to Voronoi."); effective_method = 'voronoi'
        
        if effective_method == 'none' or len(room_pins) <= 1:
            base_regions.append([_fill_holes(master_shape), " / ".join(sorted([p.name for p in room_pins])) if room_pins else f"Room_{i}"])
        elif effective_method == 'voronoi':
            base_regions.extend(_subdivide_by_voronoi(master_shape, room_pins))
        elif effective_method == 'raw':
            sub_regions, sub_names, leftover_area = _subdivide_by_raw(master_shape, room_pins, raw_rooms)
            if not sub_regions:
                print(f"      [Warning] Raw subdivision failed completely. Invoking Voronoi as fallback."); base_regions.extend(_subdivide_by_voronoi(master_shape, room_pins))
            else:
                if not leftover_area.is_empty:
                    decomposed_leftovers = list(leftover_area.geoms) if isinstance(leftover_area, MultiPolygon) else [leftover_area]
                    if merge_slivers:
                        for piece in decomposed_leftovers:
                            neighbors = _get_adjacent_regions(piece, sub_regions)
                            if neighbors: sub_regions[neighbors[0]['index']] = _fill_holes(unary_union([sub_regions[neighbors[0]['index']], piece]))
                    else: unmerged_slivers.extend(decomposed_leftovers)
                for shape, name in zip(sub_regions, sub_names):
                    base_regions.append([shape, name])
    
    return base_regions, unmerged_slivers

def _perform_merging(base_regions, pieces_to_merge):
    print("......performing final merge of geometries...")
    merger_map = {i: [] for i in range(len(base_regions))}; base_region_shapes = [r[0] for r in base_regions]
    for piece in pieces_to_merge:
        neighbors = _get_adjacent_regions(piece, base_region_shapes)
        for i in range(min(2, len(neighbors))): merger_map[neighbors[i]['index']].append(piece)
    final_layout = []
    for i, (base_shape, name) in enumerate(base_regions):
        final_shape = _fill_holes(unary_union([base_shape] + merger_map[i])) if merger_map[i] else base_shape
        final_layout.append((final_shape, name))
    return final_layout

def generate_final_vector_layout(fp: FloorPlan, merge_unlabeled: bool, subdivision_method: str):
    floor_id = list(fp.floor_plan_layouts.get("redraw").keys())[0]
    all_redraw, all_raw = fp.floor_plan_layouts["redraw"][floor_id], fp.floor_plan_layouts.get("raw", {}).get(floor_id, [])
    redraw_rooms = [to_shapely(p) for p in all_redraw if p.type == PolygonType.ROOM]
    redraw_pins = [p for p in all_redraw if p.type == PolygonType.PIN_LABEL]
    raw_rooms = [to_shapely(p) for p in all_raw if p.type == PolygonType.ROOM]
    doors = [p for p in all_redraw if p.type == PolygonType.DOOR]
    base_regions, unmerged_slivers = _subdivide_and_label_rooms(redraw_rooms, redraw_pins, raw_rooms, merge_unlabeled, subdivision_method)
    door_hulls = _find_and_pair_doorways(doors)
    if merge_unlabeled: return _perform_merging(base_regions, door_hulls), redraw_pins
    else:
        final_layout = [tuple(r) for r in base_regions]
        final_layout.extend([(hull, "Traversable") for hull in door_hulls]); final_layout.extend([(sliver, "Traversable") for sliver in unmerged_slivers])
        return final_layout, redraw_pins

# The main, rasterize, and visualize functions are unchanged and complete.
# They are included here for a fully self-contained script.
def main():
    parser = argparse.ArgumentParser(description="ZInD Advanced Grid Map Generation Tool (v-Final).", formatter_class=argparse.RawTextHelpFormatter)
    # (All arguments are assumed to be defined as in the previous working version)
    parser.add_argument("json_filepath", type=str, help="Path to the ZInD JSON file.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save the grid map (.npy) and metadata (.meta.json).")
    parser.add_argument("-r", "--resolution", type=float, default=0.05, help="Grid resolution in meters/pixel.")
    parser.add_argument("--subdivision_method", type=str, choices=['none', 'raw', 'voronoi'], default='raw', help="Algorithm for subdividing multi-pin rooms.")
    parser.add_argument("--merge_unlabeled", action="store_true", help="Merge unlabeled slivers and doorways into adjacent rooms.")
    parser.add_argument("--viz_vector", action="store_true", help="Show/save a plot of the intermediary vector data for debugging.")
    parser.add_argument("--no_viz", action="store_true", help="Suppress the final interactive grid map visualization.")
    parser.add_argument("--save_visualizations", action="store_true", help="Save visualization JPEGs to the output directory instead of showing them.")
    args = parser.parse_args()

    if not Path(args.json_filepath).is_file():
        print(f"\nError: File not found at '{Path(args.json_filepath).resolve()}'\n"); sys.exit(1)

    try:
        fp = FloorPlan(args.json_filepath)
        print("Step 1: Generating final vector layout..."); 
        final_vector_layout, all_pins = generate_final_vector_layout(fp, args.merge_unlabeled, args.subdivision_method)
        
        # --- CRITICAL FIX: Restore logic for non-interactive saving of visualizations ---
        if args.viz_vector:
            vector_save_path = None
            if args.save_visualizations:
                if not args.output_path:
                    print("\n[Error] --output_path must be specified with --save_visualizations\n"); sys.exit(1)
                output_p = Path(args.output_path)
                vector_save_path = output_p.with_name(output_p.stem + "_vector.jpeg")
            
            print(f"Step 2: Launching vector visualization..."); 
            visualize_vector_layout(final_vector_layout, all_pins, Path(args.json_filepath).name, save_path=vector_save_path)
        
        print("Step 3: Rasterizing final vector layout..."); 
        grid_map, metadata = rasterize_layout(final_vector_layout, args.resolution)
        
        if args.output_path:
            import json
            output_p = Path(args.output_path); output_p.parent.mkdir(parents=True, exist_ok=True); np.save(output_p, grid_map)
            meta_p = output_p.with_suffix('.meta.json');
            with open(meta_p, 'w') as f: json.dump(metadata, f, indent=4)
            print(f"Step 4: Saved outputs to {output_p} and {meta_p}")

        if not args.no_viz:
            raster_save_path = None
            if args.save_visualizations:
                 if not args.output_path:
                    print("\n[Error] --output_path must be specified with --save_visualizations\n"); sys.exit(1)
                 output_p = Path(args.output_path)
                 raster_save_path = output_p.with_name(output_p.stem + "_raster.jpeg")

            print("Step 5: Launching final grid map visualization..."); 
            visualize_raster_map(grid_map, metadata, Path(args.json_filepath).name, save_path=raster_save_path)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}"); 
        import traceback; traceback.print_exc(); 
        sys.exit(1)
def rasterize_layout(vector_layout, resolution):
    all_polygons = []; id_to_name_map = {}
    for shape, name in vector_layout:
        if not shape.is_empty: all_polygons.extend(list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape])
    if not all_polygons: raise ValueError("No valid geometry found.")
    all_points_stacked = np.vstack([np.array(p.exterior.coords) for p in all_polygons])
    min_coords = np.min(all_points_stacked, axis=0)-(2*resolution); max_coords = np.max(all_points_stacked, axis=0)+(2*resolution)
    grid_dims = np.ceil((max_coords - min_coords) / resolution).astype(int)
    grid_height, grid_width = grid_dims[1], grid_dims[0]
    grid_map = np.zeros((grid_height, grid_width), dtype=np.uint16); current_id = 2
    for shape, name in vector_layout:
        if name == "Traversable": region_id = 1
        else:
            region_id = current_id; id_to_name_map[region_id] = name; current_id += 1
        for poly in (list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]):
            if poly.is_empty: continue
            points_np = np.array(poly.exterior.coords); poly_grid = (points_np - min_coords) / resolution
            rr, cc = draw_polygon(poly_grid[:, 1], poly_grid[:, 0], shape=grid_map.shape); grid_map[rr, cc] = region_id
            for interior in poly.interiors:
                points_np = np.array(interior.coords); poly_grid = (points_np - min_coords) / resolution
                rr, cc = draw_polygon(poly_grid[:, 1], poly_grid[:, 0], shape=grid_map.shape); grid_map[rr, cc] = 0
    metadata = {"origin": min_coords.tolist(), "resolution": resolution, "id_to_name_map": id_to_name_map}
    return grid_map, metadata

def visualize_vector_layout(vector_layout, pins, filename, save_path=None):
    """
    Visualizes the intermediary vector data. If a save_path is provided,
    it saves the image non-interactively instead of showing a window.
    """
    # Use a non-interactive backend if we are just saving the file
    if save_path:
        plt.switch_backend('Agg')

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(f"Vector Layout Debug View for: {filename}")
    ax.set_aspect('equal', adjustable='box')
    
    unique_names = sorted(list(set(r[1] for r in vector_layout)))
    cmap = plt.get_cmap('tab20b')
    color_map = {name: cmap(i % 20) for i, name in enumerate(unique_names)}
    color_map["Traversable"] = '#cccccc'
    
    legend_handles = [patches.Patch(color=c, label=n) for n, c in color_map.items()]
    
    for shape, name in vector_layout:
        color = color_map[name]
        # Decompose MultiPolygons for plotting
        polys_to_draw = list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]
        for poly in polys_to_draw:
            if poly.is_empty: continue
            patch = patches.Polygon(list(poly.exterior.coords), closed=True, facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(patch)
            rp = poly.representative_point()
            ax.text(rp.x, rp.y, name, ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    pin_x = [p.points[0].x for p in pins]
    pin_y = [p.points[0].y for p in pins]
    ax.plot(pin_x, pin_y, 'ro', markersize=5, label='Original Pins')
    legend_handles.append(ax.get_legend_handles_labels()[0][-1]) # Add pin marker to legend
    
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    if save_path:
        print(f"......saving vector visualization to {save_path}")
        plt.savefig(save_path, format='jpeg', quality=95, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig) # Essential for batch processing
    else:
        plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

def visualize_raster_map(grid_map, metadata, filename, save_path=None):
    """
    Displays the generated grid map. If a save_path is provided, it saves
    the image to that path instead of showing an interactive window.
    """
    # Use a non-interactive backend if we are just saving the file
    if save_path:
        plt.switch_backend('Agg')

    if grid_map.size == 0:
        print("Warning: Grid map is empty, skipping visualization.")
        return
        
    fig, ax = plt.subplots(figsize=(14, 14), dpi=150)
    
    max_id = int(grid_map.max())
    
    # Use a visually pleasant, categorical colormap
    cmap = plt.get_cmap('tab20b')
    # Ensure enough colors for all unique IDs by cycling through the colormap
    num_colors = max_id + 1
    colors = cmap(np.arange(num_colors) % 20)
    
    # Assign specific, intuitive colors for special IDs
    colors[0] = [0.15, 0.15, 0.15, 1]  # Dark Gray for Obstacle (ID 0)
    if max_id >= 1:
        colors[1] = [0.9, 0.9, 0.9, 1]  # Light Gray for Traversable (ID 1)
    
    custom_cmap = ListedColormap(colors)
    
    ax.imshow(grid_map, cmap=custom_cmap, origin='lower', interpolation='none')
    
    # Draw labels on each region
    id_to_name = metadata.get("id_to_name_map", {})
    for room_id_str, name in id_to_name.items():
        room_id = int(room_id_str)
        coords = np.argwhere(grid_map == room_id)
        if len(coords) > 0:
            center = coords.mean(axis=0)
            ax.text(center[1], center[0], f"{name}\n(ID:{room_id_str})",
                    ha='center', va='center', fontsize=14,
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
    
    # Create a clear legend
    legend_patches = [
        patches.Patch(color=colors[0], label='0: Obstacle'),
        patches.Patch(color=colors[1], label='1: Traversable'),
        patches.Patch(color=cmap(0), label='2+: Unique Room ID')
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize='small')
    ax.set_title(f"Advanced Grid Map for: {filename}")
    
    # --- Final Step: Save or Show ---
    if save_path:
        print(f"......saving raster visualization to {save_path}")
        # Ensure plot is drawn without extra whitespace before saving
        plt.tight_layout(pad=0.5)
        plt.savefig(save_path, format='jpeg', quality=95, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig) # Essential for freeing memory in a batch process
    else:
        plt.tight_layout()
        plt.show()
if __name__ == '__main__':
    main()