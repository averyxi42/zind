"""
ZInD Advanced Grid Map Generation Tool (v27 - GeometryCollection Fix)

This definitive version corrects the critical `AttributeError: 'GeometryCollection'
object has no attribute 'exterior'` bug.

Key Correction:
- **Robust Geometry Handling:** A new helper function, `_extract_polygons`, is
  introduced to sanitize the output of all Shapely geometric operations. This
  function filters any `GeometryCollection` or other mixed-type results,
  ensuring that only valid `Polygon` objects are ever processed for bounding,
  rasterization, or visualization. This makes the entire pipeline robust to
  geometric edge cases that produce non-polygonal artifacts.
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
    from shapely.geometry import Polygon as ShapelyPolygon, Point as ShapelyPoint, MultiPoint, MultiPolygon, GeometryCollection
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

def _extract_polygons(shape):
    """
    CRITICAL HELPER: Takes a Shapely geometry object and returns a list of
    only its constituent Polygon objects, discarding any stray points or lines.
    """
    if shape.is_empty:
        return []
    if isinstance(shape, ShapelyPolygon):
        return [shape]
    if isinstance(shape, (MultiPolygon, GeometryCollection)):
        return [p for p in shape.geoms if isinstance(p, ShapelyPolygon) and not p.is_empty]
    return []

def _get_adjacent_regions(piece, candidates):
    neighbors = []
    for i, region in enumerate(candidates):
        if piece.buffer(1e-9).intersects(region.buffer(1e-9)):
            try:
                boundary_len = piece.intersection(region).length
                if boundary_len > 1e-6: neighbors.append({'index': i, 'boundary': boundary_len})
            except Exception: continue
    return sorted(neighbors, key=lambda x: x['boundary'], reverse=True)

import math
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

# Assume the necessary ZInD utility classes are available
# from utils import Point2D, Polygon

def _find_and_pair_doorways(doors: list) -> list:
    """
    Identifies pairs of internal doors and returns their connecting polygons.
    This version uses robust, manual polygon construction to avoid precision
    errors from ConvexHull on collinear points.
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
            # Normalize vectors to avoid floating point issues with dot product
            norm_v1 = v1 / np.linalg.norm(v1)
            norm_v2 = v2 / np.linalg.norm(v2)
            if 1.0 - abs(np.dot(norm_v1, norm_v2)) > math.sin(math.radians(10)): continue
            
            paired_indices.add(i); paired_indices.add(j)
            
            # --- DEFINITIVE FIX: Replace ConvexHull with robust manual polygon creation ---
            # The four vertices are the two points from the first door line and two from the second.
            # We order them to form a non-self-intersecting quadrilateral that defines the doorway.
            # The order is: door1_p1, door1_p2, door2_p2, door2_p1.
            doorway_poly = ShapelyPolygon([
                (p1_np[0][0], p1_np[0][1]),
                (p1_np[1][0], p1_np[1][1]),
                (p2_np[1][0], p2_np[1][1]),
                (p2_np[0][0], p2_np[0][1])
            ])
            doorway_polys.append(doorway_poly)
            break
            
    return doorway_polys

def _merge_pieces_by_voronoi(master_shape, leftover_area, sub_regions, sub_names, room_pins):
    if leftover_area.is_empty: return
    pin_points = [ShapelyPoint(p.points[0].x, p.points[0].y) for p in room_pins]
    if not pin_points: return
    try: voronoi_cells = voronoi_diagram(MultiPoint(pin_points), envelope=master_shape)
    except Exception: return

    for i, (sub_shape, sub_name) in enumerate(zip(sub_regions, sub_names)):
        parent_pin = next((p for p in room_pins if p.name == sub_name), None)
        if not parent_pin: continue
        parent_pin_point = ShapelyPoint(parent_pin.points[0].x, parent_pin.points[0].y)
        
        parent_voronoi_cell = None
        for cell in voronoi_cells.geoms:
            if cell.buffer(1e-9).contains(parent_pin_point): parent_voronoi_cell = cell; break
        
        if parent_voronoi_cell:
            piece_to_merge = leftover_area.intersection(parent_voronoi_cell)
            if not piece_to_merge.is_empty:
                sub_regions[i] = unary_union([sub_shape, piece_to_merge])

def _subdivide_by_voronoi(master_shape, room_pins):
    regions = []
    pin_points = [ShapelyPoint(p.points[0].x, p.points[0].y) for p in room_pins]
    if not pin_points: return [[master_shape, "Unlabeled_Area"]]
    try: voronoi_cells = voronoi_diagram(MultiPoint(pin_points), envelope=master_shape)
    except Exception as e:
        print(f"      [Warning] Voronoi failed: {e}. Not subdividing.");
        return [[master_shape, " / ".join(sorted([p.name for p in room_pins]))]]

    for cell in voronoi_cells.geoms:
        final_cell = master_shape.intersection(cell)
        cell_center = final_cell.representative_point()
        closest_pin_idx = min(range(len(pin_points)), key=lambda k: cell_center.distance(pin_points[k]))
        pin_name = room_pins[closest_pin_idx].name
        
        for poly in _extract_polygons(final_cell):
            regions.append([poly, pin_name])
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
            for poly in _extract_polygons(sub_region):
                successful_sub_regions.append(poly); successful_sub_names.append(pin.name)
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
    """Subdivides multi-pin rooms using the selected algorithm and hierarchical fallbacks."""
    print(f"......subdivision method: {subdivision_method.upper()}")
    pins_in_room, base_regions, unmerged_slivers = {}, [], []
    for pin in redraw_pins:
        pin_point = ShapelyPoint(pin.points[0].x, pin.points[0].y)
        for i, room_shape in enumerate(redraw_rooms):
            if room_shape.buffer(1e-9).contains(pin_point):
                pins_in_room.setdefault(i, []).append(pin); break
    
    for i, master_shape in enumerate(redraw_rooms):
        room_pins = pins_in_room.get(i, [])
        effective_method = subdivision_method
        
        if subdivision_method == 'raw' and len(room_pins) > 1:
            all_guiding_raw_polys = [s for pin in room_pins for s in raw_rooms if s.buffer(1e-9).contains(ShapelyPoint(pin.points[0].x, pin.points[0].y))]
            if all_guiding_raw_polys:
                raw_union_shape = unary_union(all_guiding_raw_polys).intersection(master_shape)
                iou = raw_union_shape.area / master_shape.area if master_shape.area > 1e-9 else 0
                if iou > RAW_FALLBACK_IOU_THRESHOLD:
                    print(f"      [Info] Raw geometry IoU > {RAW_FALLBACK_IOU_THRESHOLD}. Falling back to Voronoi for entire room.")
                    effective_method = 'voronoi'
        
        if effective_method == 'none' or len(room_pins) <= 1:
            base_regions.append([master_shape, " / ".join(sorted([p.name for p in room_pins])) if room_pins else f"Room_{i}"])
        elif effective_method == 'voronoi':
            base_regions.extend(_subdivide_by_voronoi(master_shape, room_pins))
        elif effective_method == 'raw':
            sub_regions, sub_names, leftover_area = _subdivide_by_raw(master_shape, room_pins, raw_rooms)
            if not sub_regions:
                print(f"      [Warning] Raw subdivision failed completely. Invoking Voronoi as fallback.")
                base_regions.extend(_subdivide_by_voronoi(master_shape, room_pins))
            else:
                if not leftover_area.is_empty:
                    if merge_slivers:
                        _merge_pieces_by_voronoi(master_shape, leftover_area, sub_regions, sub_names, room_pins)
                    else:
                        unmerged_slivers.extend(_extract_polygons(leftover_area))
                for shape, name in zip(sub_regions, sub_names):
                    base_regions.append([shape, name])
    
    return base_regions, unmerged_slivers

def _perform_merging(base_regions, pieces_to_merge):
    """Merges a list of polygons into a list of base regions."""
    print("......performing final merge of geometries...")
    merger_map = {i: [] for i in range(len(base_regions))}
    base_region_shapes = [r[0] for r in base_regions]
    for piece in pieces_to_merge:
        neighbors = _get_adjacent_regions(piece, base_region_shapes)
        for i in range(min(2, len(neighbors))):
            merger_map[neighbors[i]['index']].append(piece)
    final_layout = []
    for i, (base_shape, name) in enumerate(base_regions):
        final_shape = unary_union([base_shape] + merger_map[i]) if merger_map[i] else base_shape
        final_layout.append((final_shape, name))
    return final_layout

def generate_final_vector_layout(fp: FloorPlan, merge_unlabeled: bool, subdivision_method: str):
    """Orchestrates the geometric processing to produce the final vector layout."""
    floor_id = list(fp.floor_plan_layouts.get("redraw").keys())[0]
    all_redraw, all_raw = fp.floor_plan_layouts["redraw"][floor_id], fp.floor_plan_layouts.get("raw", {}).get(floor_id, [])

    redraw_rooms = [to_shapely(p) for p in all_redraw if p.type == PolygonType.ROOM]
    redraw_pins = [p for p in all_redraw if p.type == PolygonType.PIN_LABEL]
    raw_rooms = [to_shapely(p) for p in all_raw if p.type == PolygonType.ROOM]
    doors = [p for p in all_redraw if p.type == PolygonType.DOOR]

    base_regions, unmerged_slivers = _subdivide_and_label_rooms(redraw_rooms, redraw_pins, raw_rooms, merge_unlabeled, subdivision_method)
    door_hulls = _find_and_pair_doorways(doors)
    
    if merge_unlabeled:
        return _perform_merging(base_regions, door_hulls), redraw_pins
    else:
        final_layout = [tuple(r) for r in base_regions]
        final_layout.extend([(hull, "Traversable") for hull in door_hulls])
        final_layout.extend([(sliver, "Traversable") for sliver in unmerged_slivers])
        return final_layout, redraw_pins

def rasterize_layout(vector_layout, resolution):
    """Takes the final vector layout and rasterizes it into a grid map."""
    print("...rasterizing final vector layout...")
    # Sanitize the layout to get a flat list of simple polygons for bounding box calculation
    all_simple_polygons = []
    for shape, name in vector_layout:
        all_simple_polygons.extend(_extract_polygons(shape))
    if not all_simple_polygons: raise ValueError("No valid geometry found to create a map.")

    all_points_stacked = np.vstack([np.array(p.exterior.coords) for p in all_simple_polygons])
    min_coords = np.min(all_points_stacked, axis=0)-(2*resolution); max_coords = np.max(all_points_stacked, axis=0)+(2*resolution)
    grid_dims = np.ceil((max_coords - min_coords) / resolution).astype(int)
    grid_height, grid_width = grid_dims[1], grid_dims[0]
    
    grid_map = np.zeros((grid_height, grid_width), dtype=np.uint16)
    id_to_name_map, current_id = {}, 2
    for shape, name in vector_layout:
        region_id = 1 if name == "Traversable" else current_id
        if name != "Traversable":
            id_to_name_map[region_id] = name; current_id += 1
        
        # Sanitize each shape before rastering
        for poly in _extract_polygons(shape):
            points_np = np.array(poly.exterior.coords); poly_grid = (points_np - min_coords) / resolution
            rr, cc = draw_polygon(poly_grid[:, 1], poly_grid[:, 0], shape=grid_map.shape)
            grid_map[rr, cc] = region_id
            for interior in poly.interiors:
                points_np = np.array(interior.coords); poly_grid = (points_np - min_coords) / resolution
                rr, cc = draw_polygon(poly_grid[:, 1], poly_grid[:, 0], shape=grid_map.shape)
                grid_map[rr, cc] = 0 # Set holes to Obstacle
    
    metadata = {"origin": min_coords.tolist(), "resolution": resolution, "id_to_name_map": id_to_name_map}
    return grid_map, metadata

def visualize_vector_layout(vector_layout, pins, filename):
    fig, ax = plt.subplots(figsize=(14, 14)); ax.set_title(f"Vector Layout Debug View for: {filename}"); ax.set_aspect('equal', adjustable='box')
    unique_names = sorted(list(set(r[1] for r in vector_layout)))
    cmap = plt.get_cmap('tab20b'); color_map = {name: cmap(i % 20) for i, name in enumerate(unique_names)}; color_map["Traversable"] = '#cccccc'
    legend_handles = [patches.Patch(color=c, label=n) for n, c in color_map.items()]
    for shape, name in vector_layout:
        color = color_map[name]
        for poly in _extract_polygons(shape):
            patch = patches.Polygon(list(poly.exterior.coords), closed=True, facecolor=color, edgecolor='black', alpha=0.7); ax.add_patch(patch)
            rp = poly.representative_point(); ax.text(rp.x, rp.y, name, ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    pin_x = [p.points[0].x for p in pins]; pin_y = [p.points[0].y for p in pins]
    ax.plot(pin_x, pin_y, 'ro', markersize=5, label='Original Pins'); legend_handles.append(ax.get_legend_handles_labels()[0][-1])
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1)); plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout(rect=[0, 0, 0.8, 1]); plt.show()

def visualize_raster_map(grid_map, metadata, filename):
    if grid_map.size == 0: return
    fig, ax = plt.subplots(figsize=(14, 14)); max_id = int(grid_map.max()); 
    cmap = plt.get_cmap('tab20b'); num_colors = max_id + 1; colors = cmap(np.arange(num_colors) % 20)
    colors[0]=[0.15,0.15,0.15,1];
    if max_id >= 1: colors[1]=[0.9,0.9,0.9,1];
    custom_cmap = ListedColormap(colors)
    ax.imshow(grid_map, cmap=custom_cmap, origin='lower', interpolation='none'); id_to_name = metadata.get("id_to_name_map", {})
    for room_id_str, name in id_to_name.items():
        coords = np.argwhere(grid_map == int(room_id_str))
        if len(coords) > 0:
            center = coords.mean(axis=0)
            ax.text(center[1], center[0], f"{name}\n(ID:{room_id_str})", ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
    legend_patches = [patches.Patch(color=colors[0], label='0: Obstacle'), patches.Patch(color=colors[1], label='1: Traversable'), patches.Patch(color=cmap(0), label='2+: Unique Room ID')]
    ax.legend(handles=legend_patches, loc='upper right', fontsize='small'); ax.set_title(f"Advanced Grid Map for: {filename}")
    plt.tight_layout(); plt.show()

def main():
    parser = argparse.ArgumentParser(description="ZInD Advanced Grid Map Generation Tool (v27 - GeometryCollection Fix).", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("json_filepath", type=str, help="Path to the ZInD JSON file.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save outputs.")
    parser.add_argument("-r", "--resolution", type=float, default=0.05, help="Grid resolution in meters/pixel.")
    parser.add_argument("--subdivision_method", type=str, choices=['none', 'raw', 'voronoi'], default='raw', 
                        help="Algorithm for subdividing multi-pin rooms.")
    parser.add_argument("--merge_unlabeled", action="store_true", help="Merge unlabeled slivers and doorways into adjacent rooms.")
    parser.add_argument("--viz_vector", action="store_true", help="Show a plot of the intermediary vector data for debugging.")
    parser.add_argument("--no_viz", action="store_true", help="Suppress the final grid map visualization.")
    args = parser.parse_args()

    if not Path(args.json_filepath).is_file():
        print(f"\nError: File not found at '{Path(args.json_filepath).resolve()}'\n"); sys.exit(1)

    try:
        fp = FloorPlan(args.json_filepath)
        print("Step 1: Generating final vector layout..."); final_vector_layout, all_pins = generate_final_vector_layout(fp, args.merge_unlabeled, args.subdivision_method)
        if args.viz_vector:
            print("Step 2: Launching vector visualization..."); visualize_vector_layout(final_vector_layout, all_pins, Path(args.json_filepath).name)
        print("Step 3: Rasterizing final vector layout..."); grid_map, metadata = rasterize_layout(final_vector_layout, args.resolution)
        if args.output_path:
            import json
            output_filepath = Path(args.output_path); output_filepath.parent.mkdir(parents=True, exist_ok=True); np.save(output_filepath, grid_map)
            meta_filepath = output_filepath.with_suffix('.meta.json');
            with open(meta_filepath, 'w') as f: json.dump(metadata, f, indent=4)
            print(f"Step 4: Saved outputs to {output_filepath} and {meta_filepath}")
        if not args.no_viz:
            print("Step 5: Launching final grid map visualization..."); visualize_raster_map(grid_map, metadata, Path(args.json_filepath).name)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

if __name__ == '__main__':
    main()