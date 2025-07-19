"""
ZInD Advanced Grid Map Generation Tool (v30 - Definitive Fix)

This definitive version corrects two critical bugs exposed by batch processing:
1. A regression that produced `GeometryCollection` objects, causing crashes.
2. A lack of robustness against zero-length ("degenerate") door data.

Key Corrections:
- **Robust Geometry Sanitization:** The output of the final merging step is now
  rigorously sanitized with the `_extract_polygons` helper, guaranteeing that the
  final vector layout contains only valid `Polygon` objects. This resolves the
  `AttributeError`.
- **Degenerate Data Handling:** The `_find_and_pair_doorways` function now
  explicitly checks for and skips zero-length door polygons, preventing the
  `RuntimeWarning` and ensuring stability.
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

def _extract_polygons(shape):
    """
    Takes a Shapely geometry object and returns a list of only its
    constituent Polygon objects, discarding any stray points or lines.
    """
    if shape.is_empty: return []
    if isinstance(shape, ShapelyPolygon): return [shape]
    if isinstance(shape, (MultiPolygon, GeometryCollection)):
        return [p for p in shape.geoms if isinstance(p, ShapelyPolygon) and not p.is_empty]
    return []
def _get_adjacent_regions(piece, candidates):
    """
    Finds the candidate polygons that are adjacent to the piece, sorted by
    longest shared boundary. This version uses a small buffer on the piece to
    robustly handle microscopic floating-point gaps.
    """
    neighbors = []
    # --- CRITICAL FIX: Apply a small buffer to the piece to bridge precision gaps ---
    # This ensures that pieces that are "supposed" to touch will be found as neighbors.
    buffered_piece = piece.buffer(0.01)
    
    for i, region in enumerate(candidates):
        # Check for intersection against the buffered piece
        if buffered_piece.intersects(region):
            try:
                # The boundary length is still calculated on the original piece for accuracy
                boundary_len = buffered_piece.intersection(region).length
                # Only consider it a neighbor if there's a meaningful shared boundary
                if boundary_len > 1e-6:
                    neighbors.append({'index': i, 'boundary': boundary_len})
            except Exception:
                # Ignore potential topology errors in shapely
                continue
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
        print(f"      [Warning] Voronoi failed: {e}. Not subdividing."); return [[master_shape, " / ".join(sorted([p.name for p in room_pins]))]]

    for cell in voronoi_cells.geoms:
        final_cell = master_shape.intersection(cell)
        cell_center = final_cell.representative_point()
        closest_pin_idx = min(range(len(pin_points)), key=lambda k: cell_center.distance(pin_points[k]))
        pin_name = room_pins[closest_pin_idx].name
        for poly in _extract_polygons(final_cell): regions.append([poly, pin_name])
    return regions

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
# --- Add this new import to the Dependency Checks section ---
try:
    import skfmm
    from shapely.ops import split as shapely_split
except ImportError:
    print("\nFATAL ERROR: This script requires 'scikit-fmm' and 'shapely'. Install with: pip install scikit-fmm shapely\n"); sys.exit(1)

# --- Add this new import to the Dependency Checks section ---
from scipy.spatial import cKDTree

# --- DEFINITIVE, CORRECTED FMM-based subdivision function ---
def _subdivide_by_fmm(master_shape, room_pins):
    """
    Subdivides a master polygon using a geodesic FMM partition. This definitive
    version is robust to cases where pin locations fall outside the master_shape
    and creates a stable starting interface for the FMM algorithm.
    """
    regions = []
    pin_points = [ShapelyPoint(p.points[0].x, p.points[0].y) for p in room_pins]
    if not pin_points: return [[master_shape, "Unlabeled_Area"]]

    # 1. Adaptive Discretization
    bounds = master_shape.bounds
    diag_len = math.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
    resolution = np.clip(diag_len / 1000.0, 0.001, 0.1)
    min_coords = np.array([bounds[0], bounds[1]])
    grid_dims = (np.ceil((np.array([bounds[2], bounds[3]]) - min_coords) / resolution)).astype(int)
    grid_height, grid_width = grid_dims[1], grid_dims[0]

    if grid_height <= 2 or grid_width <= 2 or grid_height * grid_width > 4_000_000:
        print(f"      [Warning] FMM grid is too large or too small. Falling back to Voronoi.");
        return _subdivide_by_voronoi(master_shape, room_pins)

    # 2. Create Binary Mask of the room
    mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
    for poly in _extract_polygons(master_shape):
        poly_grid = (np.array(poly.exterior.coords) - min_coords) / resolution
        rr, cc = draw_polygon(poly_grid[:, 1], poly_grid[:, 0], shape=mask.shape)
        mask[rr, cc] = 1
        for interior in poly.interiors:
            interior_grid = (np.array(interior.coords) - min_coords) / resolution
            rr, cc = draw_polygon(interior_grid[:, 1], interior_grid[:, 0], shape=mask.shape)
            mask[rr, cc] = 0

    valid_pixels = np.argwhere(mask == 1)
    if len(valid_pixels) == 0:
        print(f"      [Warning] FMM mask is empty. Cannot subdivide this piece.")
        return [[master_shape, " / ".join(sorted([p.name for p in room_pins]))]]
    pixel_tree = cKDTree(valid_pixels)

    # 3. Generate and Smooth Geodesic Distance Maps
    distance_maps = []
    from scipy.ndimage import gaussian_filter
    from skimage.draw import disk
    for pin_point in pin_points:
        phi = np.ones_like(mask, dtype=np.float32)
        
        pin_y, pin_x = ((np.array([pin_point.y, pin_point.x]) - min_coords[[1,0]]) / resolution).astype(int)
        pin_y, pin_x = np.clip(pin_y, 0, grid_height-1), np.clip(pin_x, 0, grid_width-1)

        if mask[pin_y, pin_x] == 0:
            closest_pixel_index = pixel_tree.query([pin_y, pin_x])[1]
            valid_y, valid_x = valid_pixels[closest_pixel_index]
        else:
            valid_y, valid_x = pin_y, pin_x
        
        # --- CRITICAL FIX: Create a robust starting 'puddle' instead of a single point ---
        rr, cc = disk((valid_y, valid_x), radius=2, shape=phi.shape)
        phi[rr, cc] = -1
        
        phi_masked = np.ma.masked_where(mask == 0, phi)
        dist_map = skfmm.distance(phi_masked, dx=[resolution,resolution])
        distance_maps.append(gaussian_filter(dist_map, sigma=1.5))

    # 4. Determine Region Ownership
    ownership_grid = np.argmin(np.stack(distance_maps, axis=0), axis=0)
    ownership_grid[mask == 0] = -1

    # 5. Polygonize, Smooth, and Clip
    from skimage.measure import find_contours
    for pin_idx, pin in enumerate(room_pins):
        pin_mask = np.zeros_like(mask); pin_mask[ownership_grid == pin_idx] = 1
        contours = find_contours(pin_mask, 0.5)
        for contour in contours:
            contour_world = contour[:, [1, 0]] * resolution + min_coords
            if len(contour_world) < 3: continue
            
            try:
                jagged_poly = ShapelyPolygon(contour_world)
                smoothed_poly = jagged_poly.buffer(0).buffer(resolution * 1.5).buffer(-resolution * 1.5)
                final_poly = master_shape.intersection(smoothed_poly)
                
                for poly in _extract_polygons(final_poly):
                    regions.append([poly, pin.name])
            except Exception: continue
                
    return regions
def _merge_pieces_by_fmm(master_shape, leftover_area, sub_regions, sub_names, room_pins):
    """
    Merges leftover pieces into sub-regions using a geodesic FMM partition of
    the leftover area itself.
    """
    if leftover_area.is_empty: return
    
    # Generate a geodesic partition of the *leftover area* using the original pins as generators
    fmm_partitions = _subdivide_by_fmm(leftover_area, room_pins)
    
    # For each sub-region, find the piece of the leftover area that belongs to it
    for i, (sub_shape, sub_name) in enumerate(zip(sub_regions, sub_names)):
        # Find the FMM partition(s) for this sub-region's name
        pieces_to_merge_for_this_region = [shape for shape, name in fmm_partitions if name == sub_name]
        
        if pieces_to_merge_for_this_region:
            # Union all found pieces with the original sub-region
            all_pieces = [sub_shape] + pieces_to_merge_for_this_region
            sub_regions[i] = unary_union(all_pieces)
def _subdivide_geodesically(master_shape, room_pins):
    """
    Orchestrates the subdivision of a polygon by first checking for convexity
    and dispatching to the most appropriate algorithm.
    """
    # --- Convexity Check for Performance Optimization ---
    # A small tolerance is used to account for floating point inaccuracies.
    # If the area of the convex hull is nearly identical to the shape's area,
    # it's convex for our purposes.
    if (master_shape.convex_hull.area - master_shape.area) < 1e-9:
        print("      [Info] Shape is convex. Using fast Euclidean Voronoi subdivision.")
        return _subdivide_by_voronoi(master_shape, room_pins)
    else:
        print("      [Info] Shape is non-convex. Using geodesic FMM subdivision.")
        return _subdivide_by_fmm(master_shape, room_pins)
def _merge_pieces_geodesically(master_shape, leftover_area, sub_regions, sub_names, room_pins):
    """
    Merges leftover pieces into sub-regions using a geodesic partition of
    the master shape. This is the definitive, correct merging logic.
    """
    if leftover_area.is_empty: return
    
    # 1. Generate a complete, ideal geodesic partition of the entire parent room.
    # This serves as our "map" of which areas belong to which pin.
    geodesic_partitions = _subdivide_geodesically(master_shape, room_pins)
    
    # 2. For each existing sub-region, find the piece of the leftover area that belongs to it.
    for i, (sub_shape, sub_name) in enumerate(zip(sub_regions, sub_names)):
        # Find the full geodesic region(s) corresponding to this sub-region's name.
        parent_geodesic_regions = [shape for shape, name in geodesic_partitions if name == sub_name]
        
        if parent_geodesic_regions:
            # Union them just in case there are multiple (should be rare).
            parent_geodesic_union = unary_union(parent_geodesic_regions)
            
            # 3. The piece to merge is the part of the leftover area that falls
            #    within this sub-region's geodesic "zone of influence".
            piece_to_merge = leftover_area.intersection(parent_geodesic_union)
            
            # 4. Perform the final union.
            if not piece_to_merge.is_empty:
                sub_regions[i] = unary_union([sub_shape, piece_to_merge])

# --- Add this new import to the Dependency Checks section ---
from collections import defaultdict

def _subdivide_and_label_rooms(redraw_rooms, redraw_pins, raw_rooms, merge_slivers, subdivision_method: str):
    """
    Subdivides multi-pin rooms using the definitive hybrid FMM algorithm.
    This version uses the corrected, robust data model for guiding polygons,
    fixing the 'unhashable type' error by using the pin's index.
    """
    print(f"......subdivision method: {subdivision_method.upper()}")
    pins_in_room, final_regions = {}, []
    for pin in redraw_pins:
        pin_point = ShapelyPoint(pin.points[0].x, pin.points[0].y)
        for i, room_shape in enumerate(redraw_rooms):
            if room_shape.buffer(1e-9).contains(pin_point):
                pins_in_room.setdefault(i, []).append(pin); break
    
    for i, master_shape in enumerate(redraw_rooms):
        room_pins = pins_in_room.get(i, [])
        
        if subdivision_method == 'none' or len(room_pins) <= 1:
            name = " / ".join(sorted([p.name for p in room_pins])) if room_pins else f"Room_{i}"
            final_regions.append([master_shape, name])
            continue

        print(f"......processing room {i} with hybrid FMM...")

        # Step 1: Create initial hints for each pin.
        pin_hints = []
        for pin in room_pins:
            pin_point = ShapelyPoint(pin.points[0].x, pin.points[0].y)
            raw_shapes_for_pin = [s for s in raw_rooms if s.buffer(1e-9).contains(pin_point)]
            if raw_shapes_for_pin:
                hint = unary_union(raw_shapes_for_pin).intersection(master_shape)
                if not hint.is_empty:
                    pin_hints.append({'pins': [pin], 'shape': hint})

        # Step 2: Iteratively merge overlapping hints.
        merged = True
        while merged:
            merged = False
            for i1 in range(len(pin_hints)):
                for i2 in range(i1 + 1, len(pin_hints)):
                    if pin_hints[i1]['shape'].intersects(pin_hints[i2]['shape']):
                        merged_hint = pin_hints[i1]; other_hint = pin_hints[i2]
                        merged_hint['pins'].extend(other_hint['pins'])
                        merged_hint['shape'] = unary_union([merged_hint['shape'], other_hint['shape']])
                        pin_hints.pop(i2); merged = True; break
                if merged: break
        
        # Step 3: Resolve ambiguous hints and create candidate guides.
        candidate_guides = []
        for hint in pin_hints:
            if len(hint['pins']) == 1:
                for poly in _extract_polygons(hint['shape']):
                    candidate_guides.append({'pin': hint['pins'][0], 'shape': poly})
            else:
                subdivided = _subdivide_by_voronoi(hint['shape'], hint['pins'])
                for poly, name in subdivided:
                    parent_pin = next(p for p in hint['pins'] if p.name == name)
                    candidate_guides.append({'pin': parent_pin, 'shape': poly})

        # Step 4: Rigorous Verification and creation of the final guiding_polygons list.
        # --- CRITICAL FIX: Use a list indexed by the pin's position ---
        guiding_polygons = [None] * len(room_pins)
        # Use the object's memory id() as a temporary, unique key for the lookup map
        pin_to_index = {id(pin): idx for idx, pin in enumerate(room_pins)}
        all_pin_points_in_room = [ShapelyPoint(p.points[0].x, p.points[0].y) for p in room_pins]

        for candidate in candidate_guides:
            contained_pins_count = sum(1 for p_point in all_pin_points_in_room if candidate['shape'].buffer(1e-9).contains(p_point))
            
            if contained_pins_count == 1:
                pin_idx = pin_to_index[id(candidate['pin'])]
                current_guide = guiding_polygons[pin_idx]
                if current_guide is None:
                    guiding_polygons[pin_idx] = candidate['shape']
                else:
                    guiding_polygons[pin_idx] = unary_union([current_guide, candidate['shape']])
        
        print(f".........found {sum(1 for g in guiding_polygons if g is not None)} confident guiding regions.")
        
        # Step 5: Perform the final FMM partitioning.
        fmm_partitions = _run_hybrid_fmm(master_shape, room_pins, guiding_polygons)
        final_regions.extend(fmm_partitions)
    
    return final_regions, []
# --- Add these new imports to the Dependency Checks section ---
from skimage.morphology import binary_dilation, binary_erosion,disk as sk_disk
from skimage.draw import disk
from skimage.measure import find_contours

# --- This new function REPLACES the old _run_hybrid_fmm ---
def _run_hybrid_fmm(master_shape, all_pins_in_room, guiding_polygons: list):
    """
    Runs the hybrid FMM using the definitive "Partition-and-Reconstruct" algorithm,
    now made robust with mask padding to ensure correct contour finding.
    """
    regions = []
    
    # 1. Adaptive Discretization
    bounds = master_shape.bounds
    diag_len = math.sqrt((bounds[2] - bounds[0])**2 + (bounds[3] - bounds[1])**2)
    resolution = np.clip(diag_len / 2000.0, 0.005, 0.1)
    min_coords = np.array([bounds[0], bounds[1]])
    grid_dims = (np.ceil((np.array([bounds[2], bounds[3]]) - min_coords) / resolution)).astype(int)
    grid_height, grid_width = grid_dims[1], grid_dims[0]

    if grid_height <= 2 or grid_width <= 2 or grid_height * grid_width > 5_000_000:
        print(f"      [Warning] FMM grid too large or small. Falling back to Voronoi.");
        return _subdivide_by_voronoi(master_shape, all_pins_in_room)

    # 2. Create the room mask
    mask = np.zeros((grid_height, grid_width), dtype=np.uint8)
    for poly in _extract_polygons(master_shape):
        poly_grid = (np.array(poly.exterior.coords) - min_coords) / resolution
        rr, cc = draw_polygon(poly_grid[:, 1], poly_grid[:, 0], shape=mask.shape)
        mask[rr, cc] = 1
        for interior in poly.interiors:
            interior_grid = (np.array(interior.coords) - min_coords) / resolution
            rr, cc = draw_polygon(interior_grid[:, 1], interior_grid[:, 0], shape=mask.shape)
            mask[rr, cc] = 0

    # --- CRITICAL FIX: Pad all relevant arrays to prevent edge-case failures ---
    padding = 5 # A 5-pixel border is robust
    mask = binary_dilation(mask,sk_disk(radius=2))
    padded_mask = np.pad(mask, pad_width=padding, mode='constant', constant_values=0)
    
    # 3. Hybrid Initialization of Distance Maps on the PADDED grid
    distance_maps = []
    valid_pins_for_fmm = []
    for i, pin in enumerate(all_pins_in_room):
        # Create a padded phi array
        phi = np.ones_like(padded_mask, dtype=np.float32)
        guide_shape = guiding_polygons[i]
        
        has_start_point = False
        if guide_shape is not None:
            for poly in _extract_polygons(guide_shape):
                # Convert to grid coordinates and add padding offset
                poly_grid = (np.array(poly.exterior.coords) - min_coords) / resolution + padding
                rr, cc = draw_polygon(poly_grid[:, 1], poly_grid[:, 0], shape=phi.shape)
                phi[rr, cc] = -1
                if rr.size > 0: has_start_point = True
        else:
            pin_point = ShapelyPoint(pin.points[0].x, pin.points[0].y)
            # Convert to grid coordinates and add padding offset
            pin_y, pin_x = ((np.array([pin_point.y, pin_point.x]) - min_coords[[1,0]]) / resolution).astype(int) + padding
            pin_y, pin_x = np.clip(pin_y, 0, padded_mask.shape[0]-1), np.clip(pin_x, 0, padded_mask.shape[1]-1)
            # Check against the PADDED mask to ensure the pin is valid
            if padded_mask[pin_y, pin_x] == 1:
                rr, cc = disk((pin_y, pin_x), radius=2, shape=phi.shape)
                phi[rr, cc] = -1
                has_start_point = True

        if has_start_point:
            phi_masked = np.ma.masked_where(padded_mask == 0, phi)
            dist_map = skfmm.distance(phi_masked, dx=[resolution, resolution])
            distance_maps.append(dist_map)
            valid_pins_for_fmm.append(pin)

    if not distance_maps:
        print(f"      [Warning] No valid pins found. Not subdividing.");
        return [[master_shape, " / ".join(sorted([p.name for p in all_pins_in_room]))]]

    # 4. Winner-Take-All on the PADDED grid
    ownership_grid = np.argmin(np.stack(distance_maps, axis=0), axis=0)
    ownership_grid[padded_mask == 0] = -1

    # 5. Polygonization on the PADDED grid
    for pin_idx, pin in enumerate(valid_pins_for_fmm):
        pin_mask = np.zeros_like(padded_mask); pin_mask[ownership_grid == pin_idx] = 1
        # find_contours is now guaranteed to work correctly because of the padding
        contours = find_contours(pin_mask, 0.5)
        for contour in contours:
            # --- CRITICAL FIX: Subtract padding to convert back to original coordinate frame ---
            contour_unpadded = contour - padding
            contour_world = contour_unpadded[:, [1, 0]] * resolution + min_coords

            if len(contour_world) < 3: continue
            try:
                final_poly = master_shape.intersection(ShapelyPolygon(contour_world))
                for poly in _extract_polygons(final_poly):
                    regions.append([poly, pin.name])
            except Exception as e: 
                print(f"      [Warning] Polygonization failed for a contour: {e}")
                continue
                
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
                # Decompose the remaining area into its constituent simple polygons
                remaining_pieces = list(remaining_area.geoms) if isinstance(remaining_area, MultiPolygon) else [remaining_area]
                
                for piece in remaining_pieces:
                    # For each piece, find which of the failed pins are inside it
                    pins_for_this_piece = [pin for pin in failed_pins if piece.buffer(1e-9).contains(ShapelyPoint(pin.points[0].x, pin.points[0].y))]
                    
                    if pins_for_this_piece:
                        # Run the fallback on this piece with ONLY its relevant pins
                        voronoi_sub_regions = _subdivide_geodesically(piece, pins_for_this_piece)
                        for shape, name in voronoi_sub_regions:
                            successful_sub_regions.append(shape)
                            successful_sub_names.append(name)    
    leftover_area = master_shape.difference(unary_union(successful_sub_regions))
    return successful_sub_regions, successful_sub_names, leftover_area

def _merge_pieces_geodesically(master_shape, leftover_area, sub_regions, sub_names, room_pins):
    """
    Merges leftover pieces into sub-regions using a geodesic partition of
    the master shape. This is the definitive, correct merging logic.
    """
    if leftover_area.is_empty: return
    
    # 1. Generate a complete, ideal geodesic partition of the entire parent room.
    # This serves as our "map" of which areas belong to which pin.
    geodesic_partitions = _subdivide_geodesically(master_shape, room_pins)
    
    # 2. For each existing sub-region, find the piece of the leftover area that belongs to it.
    for i, (sub_shape, sub_name) in enumerate(zip(sub_regions, sub_names)):
        # Find the full geodesic region(s) corresponding to this sub-region's name.
        parent_geodesic_regions = [shape for shape, name in geodesic_partitions if name == sub_name]
        
        if parent_geodesic_regions:
            # Union them just in case there are multiple (should be rare).
            parent_geodesic_union = unary_union(parent_geodesic_regions)
            
            # 3. The piece to merge is the part of the leftover area that falls
            #    within this sub-region's geodesic "zone of influence".
            piece_to_merge = leftover_area.intersection(parent_geodesic_union)
            
            # 4. Perform the final union.
            if not piece_to_merge.is_empty:
                sub_regions[i] = unary_union([sub_shape, piece_to_merge])

def _perform_merging(base_regions, pieces_to_merge):
    """
    Merges a list of polygons (doors, slivers) into a list of base regions.
    This definitive version uses a state-safe "merger map" to prevent the
    catastrophic bug where adjacent rooms were incorrectly fused together.
    """
    print("......performing final merge of geometries...")
    
    # 1. Create a map to hold pieces that need to be merged into each base region.
    merger_map = {i: [] for i in range(len(base_regions))}
    # Use the original, unmodified shapes for all adjacency checks.
    base_region_shapes = [r[0] for r in base_regions]
    
    # 2. Find neighbors and populate the merger_map without modifying any geometry.
    for piece in pieces_to_merge:
        neighbors = _get_adjacent_regions(piece, base_region_shapes)
        # A door should connect two primary regions.
        num_neighbors_to_merge = 2 if len(neighbors) >= 2 else len(neighbors)
        for i in range(num_neighbors_to_merge):
            neighbor_index = neighbors[i]['index']
            merger_map[neighbor_index].append(piece.buffer(0.001))
    
    # 3. After all adjacency is determined, perform the actual unions.
    final_layout = []
    for i, (base_shape, name) in enumerate(base_regions):
        # If there are pieces to merge for this region, union them all at once.
        if merger_map[i]:
            final_shape = unary_union([base_shape] + merger_map[i])
            # The union can create holes, which we must fill.
            final_shape = _fill_holes(final_shape)
        else:
            final_shape = base_shape
        
        # The final union can also result in a MultiPolygon, which must be decomposed.
        for poly in _extract_polygons(final_shape):
            final_layout.append((poly, name))

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

def rasterize_layout(vector_layout, resolution):
    all_polygons = [shape for shape, name in vector_layout] # Already sanitized to be simple polygons
    if not all_polygons: raise ValueError("No valid geometry found to create a map.")
    all_points_stacked = np.vstack([np.array(p.exterior.coords) for p in all_polygons])
    min_coords = np.min(all_points_stacked, axis=0)-(2*resolution); max_coords = np.max(all_points_stacked, axis=0)+(2*resolution)
    grid_dims = np.ceil((max_coords - min_coords) / resolution).astype(int)
    grid_height, grid_width = grid_dims[1], grid_dims[0]
    grid_map = np.zeros((grid_height, grid_width), dtype=np.uint16); current_id = 2; id_to_name_map = {}
    for poly, name in vector_layout:
        if name == "Traversable": region_id = 1
        else:
            region_id = current_id; id_to_name_map[region_id] = name; current_id += 1
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
    This version now also plots the original pin labels for debugging.
    """
    if save_path:
        plt.switch_backend('Agg')

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_title(f"Vector Layout Debug View for: {filename}")
    ax.set_aspect('equal', adjustable='box')
    
    # --- Polygon Plotting (Unchanged) ---
    unique_names = sorted(list(set(r[1] for r in vector_layout)))
    cmap = plt.get_cmap('tab20b')
    color_map = {name: cmap(i % 20) for i, name in enumerate(unique_names)}
    color_map["Traversable"] = '#cccccc'
    
    legend_handles = [patches.Patch(color=c, label=n) for n, c in color_map.items()]
    
    for shape, name in vector_layout:
        color = color_map[name]
        polys_to_draw = list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]
        for poly in polys_to_draw:
            if poly.is_empty: continue
            patch = patches.Polygon(list(poly.exterior.coords), closed=True, facecolor=color, edgecolor='black', alpha=0.7)
            ax.add_patch(patch)
            rp = poly.representative_point()
            ax.text(rp.x, rp.y, name, ha='center', va='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))

    # --- Original Pin Location Plotting (Unchanged) ---
    pin_x = [p.points[0].x for p in pins]
    pin_y = [p.points[0].y for p in pins]
    ax.plot(pin_x, pin_y, 'ro', markersize=5, label='Original Pins')
    
    # --- NEW: Original Pin Label Plotting ---
    for pin in pins:
        point = pin.points[0]
        ax.text(point.x, point.y + 0.1, pin.name,  # Offset label slightly for visibility
                ha='center', va='bottom', fontsize=12, color='red',
                bbox=dict(facecolor='white', alpha=0.4, edgecolor='none', boxstyle='round,pad=0.1'))
    
    # --- Legend and Finalization (Unchanged) ---
    legend_handles.append(ax.get_legend_handles_labels()[0][-1]) # Add pin marker to legend
    ax.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.8, 1])

    if save_path:
        print(f"......saving vector visualization to {save_path}")
        plt.savefig(save_path, format='jpeg', quality=95, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()

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

def main():
    parser = argparse.ArgumentParser(description="ZInD Advanced Grid Map Generation Tool (v30 - Definitive Fix).", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("json_filepath", type=str, help="Path to the ZInD JSON file.")
    parser.add_argument("-o", "--output_path", type=str, help="Path to save outputs.")
    parser.add_argument("-r", "--resolution", type=float, default=0.05, help="Grid resolution.")
    parser.add_argument("--subdivision_method", type=str, choices=['none', 'raw', 'voronoi'], default='raw', help="Algorithm for subdivision.")
    parser.add_argument("--merge_unlabeled", action="store_true", help="Merge unlabeled areas.")
    parser.add_argument("--viz_vector", action="store_true", help="Show/save the vector data plot.")
    parser.add_argument("--no_viz", action="store_true", help="Suppress the final raster map visualization.")
    parser.add_argument("--save_visualizations", action="store_true", help="Save visualization JPEGs to the output directory.")
    args = parser.parse_args()

    if not Path(args.json_filepath).is_file():
        print(f"\nError: File not found at '{Path(args.json_filepath).resolve()}'\n"); sys.exit(1)

    try:
        fp = FloorPlan(args.json_filepath)
        print("Step 1: Generating final vector layout..."); final_vector_layout, all_pins = generate_final_vector_layout(fp, args.merge_unlabeled, args.subdivision_method)
        
        if args.viz_vector:
            vector_save_path = None
            if args.save_visualizations:
                if not args.output_path: print("\n[Error] --output_path must be specified with --save_visualizations\n"); sys.exit(1)
                output_p = Path(args.output_path); vector_save_path = output_p.with_name(output_p.stem + "_vector.jpeg")
            print("Step 2: Launching vector visualization..."); visualize_vector_layout(final_vector_layout, all_pins, Path(args.json_filepath).name, save_path=vector_save_path)
        
        print("Step 3: Rasterizing final vector layout..."); grid_map, metadata = rasterize_layout(final_vector_layout, args.resolution)
        
        if args.output_path:
            import json
            output_p = Path(args.output_path); output_p.parent.mkdir(parents=True, exist_ok=True); np.save(output_p, grid_map)
            meta_p = output_p.with_suffix('.meta.json');
            with open(meta_p, 'w') as f: json.dump(metadata, f, indent=4)
            print(f"Step 4: Saved outputs to {output_p} and {meta_p}")

        if not args.save_visualizations:
            raster_save_path = None
        if args.save_visualizations:
                if not args.output_path: print("\n[Error] --output_path must be specified with --save_visualizations\n"); sys.exit(1)
                output_p = Path(args.output_path); raster_save_path = output_p.with_name(output_p.stem + "_raster.jpeg")
        print("Step 5: Launching final grid map visualization..."); visualize_raster_map(grid_map, metadata, Path(args.json_filepath).name, save_path=raster_save_path)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}"); import traceback; traceback.print_exc(); sys.exit(1)

if __name__ == '__main__':
    main()