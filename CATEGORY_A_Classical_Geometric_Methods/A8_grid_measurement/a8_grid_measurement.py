import cv2
import numpy as np
import argparse
from utils.camera_utils import load_image, save_image

def merge_rectangles(rects, overlap_thresh=0.15, gap_thresh=15):
    if not rects:
        return []

    merged = []
    rects = [tuple(r) for r in rects]

    while rects:
        x, y, w, h = rects.pop(0)
        x2, y2 = x + w, y + h
        merged_rect = [x, y, x2, y2]
        changed = True

        while changed:
            changed = False
            i = 0
            while i < len(rects):
                rx, ry, rw, rh = rects[i]
                rx2, ry2 = rx + rw, ry + rh

                ix1, iy1 = max(merged_rect[0], rx), max(merged_rect[1], ry)
                ix2, iy2 = min(merged_rect[2], rx2), min(merged_rect[3], ry2)
                inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                area_a = (merged_rect[2] - merged_rect[0]) * (merged_rect[3] - merged_rect[1])
                area_b = rw * rh
                union = area_a + area_b - inter_area
                iou = inter_area / union if union > 0 else 0

                close_horiz = abs(rx - merged_rect[2]) <= gap_thresh or abs(merged_rect[0] - rx2) <= gap_thresh
                close_vert = abs(ry - merged_rect[3]) <= gap_thresh or abs(merged_rect[1] - ry2) <= gap_thresh

                if iou > overlap_thresh or (close_horiz and close_vert):
                    merged_rect[0] = min(merged_rect[0], rx)
                    merged_rect[1] = min(merged_rect[1], ry)
                    merged_rect[2] = max(merged_rect[2], rx2)
                    merged_rect[3] = max(merged_rect[3], ry2)
                    rects.pop(i)
                    changed = True
                else:
                    i += 1

        merged.append((merged_rect[0], merged_rect[1], merged_rect[2] - merged_rect[0], merged_rect[3] - merged_rect[1]))

    return merged


def measure_with_grid_calibration(image_path, cell_size_cm, output_path=None):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pre-processing to emphasize edges and suppress grid lines.
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        7,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Use Canny for grid calibration and a dilated cleaned mask for object detection.
    edges = cv2.Canny(blurred, 50, 150)
    grid_contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    object_mask = cv2.dilate(cleaned, dilate_kernel, iterations=2)
    object_contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 1. Collect all areas to determine what a "Grid Tile" looks like
    all_areas = [cv2.contourArea(cnt) for cnt in grid_contours if cv2.contourArea(cnt) > 100]
    if not all_areas:
        print("Error: No contours detected for grid calibration.")
        return
    
    # The grid tiles will be the most numerous, smaller objects. 
    # We find the median area to serve as our "reference tile"
    median_tile_area = np.median(all_areas)
    print(f"Calibration: Detected median grid tile area as {median_tile_area:.0f} pixels.")
    
    # 2. Set threshold: Object must be significantly larger than a grid tile
    # We define 'object' as anything > 5x the area of a single tile
    object_threshold = median_tile_area * 5
    
    # Establish pixels_per_cm using the grid tile (assuming square grid)
    # sqrt(area) gives us the side length in pixels
    tile_side_px = np.sqrt(median_tile_area)
    pixels_per_cm = tile_side_px / cell_size_cm
    
    print(f"Calibration: Pixels-per-cm = {pixels_per_cm:.2f}")
    print(f"Scanning for objects larger than {object_threshold:.0f} pixels...")

    annotated = img.copy()
    objects_found = 0

    candidate_rects = []
    for cnt in object_contours:
        area = cv2.contourArea(cnt)

        # Only process objects significantly larger than the grid reference
        if area <= object_threshold:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # Simple rectangle filter to avoid detecting the image border
        if w >= img.shape[1] * 0.9 or h >= img.shape[0] * 0.9:
            continue

        rect_area = float(w * h)
        if rect_area <= 0:
            continue

        fill_ratio = area / rect_area
        if fill_ratio < 0.25:
            # Ignore hollow outlines and loose grid-line clusters.
            continue

        # Ignore very thin/skewed contours that are unlikely to be the main object.
        if w < 0.5 * pixels_per_cm or h < 0.5 * pixels_per_cm:
            continue

        candidate_rects.append((x, y, w, h))

    merged_rects = merge_rectangles(candidate_rects)

    for x, y, w, h in merged_rects:
        real_w = w / pixels_per_cm
        real_h = h / pixels_per_cm

        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(annotated, f"{real_w:.1f}x{real_h:.1f}cm", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print(f" -> MATCH! Object detected: {real_w:.1f}x{real_h:.1f}cm")
        objects_found += 1

    if output_path:
        save_image(annotated, output_path)
    
    return {"status": "success", "processed_objects": objects_found}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A8 - Auto-Calibrating Grid Measurement")
    parser.add_argument("--image", required=True)
    parser.add_argument("--cell-size", type=float, required=True, help="Real size of one grid square in cm")
    parser.add_argument("--output", default="a8_result.jpg")
    args = parser.parse_args()
    
    print(measure_with_grid_calibration(args.image, args.cell_size, args.output))