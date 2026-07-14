"""
A6 - Contour Geometry Method
==============================
Math:
    Measures properties of irregular shapes by using a reference object 
    for scale. 

    Conversions:
        real_length = pixel_length / pixels_per_metric
        real_area   = pixel_area   / (pixels_per_metric ** 2)

    Key features:
        - cv2.minAreaRect: Best for object dimensions (rotated).
        - cv2.contourArea: Best for true surface area.
        - cv2.arcLength: Best for true object perimeter.
"""

import cv2
import json
import argparse
from utils.camera_utils import (
    load_image, find_contours, min_area_rect_dimensions,
    draw_contour, save_image, COLOR_REFERENCE
)

def analyze_contour(contour, pixels_per_metric):
    """Computes real-world geometric properties for a shape."""
    _, width_px, height_px = min_area_rect_dimensions(contour)
    perimeter_px = cv2.arcLength(contour, True)
    area_px = cv2.contourArea(contour)

    return {
        "width": width_px / pixels_per_metric,
        "height": height_px / pixels_per_metric,
        "perimeter": perimeter_px / pixels_per_metric,
        "area": area_px / (pixels_per_metric ** 2),
    }

def measure_shapes(image_path, ref_dim_real, ref_index=0, output_path=None):
    img = load_image(image_path)
    contours = find_contours(img)
    
    # Sort contours by size to ensure consistent indexing
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if len(contours) < 2:
        raise ValueError("Need at least 2 shapes: 1 ref + 1 target.")

    _, ref_width_px, _ = min_area_rect_dimensions(contours[ref_index])
    pixels_per_metric = ref_width_px / ref_dim_real

    annotated = img.copy() if output_path else None
    
    results = []
    for i, c in enumerate(contours):
        props = analyze_contour(c, pixels_per_metric)
        results.append(props)
        
        if annotated is not None:
            color = COLOR_REFERENCE if i == ref_index else (0, 220, 0)
            label = f"{props['width']:.1f}x{props['height']:.1f} A={props['area']:.1f}"
            draw_contour(annotated, c, color=color, label=label)

    if output_path:
        save_image(annotated, output_path)

    return {"pixels_per_metric": pixels_per_metric, "shapes": results}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A6 - Contour Geometry")
    parser.add_argument("--image", required=True)
    parser.add_argument("--ref-width", type=float, required=True)
    parser.add_argument("--output", default="a6_result.jpg")
    
    args = parser.parse_args()
    res = measure_shapes(args.image, args.ref_width, output_path=args.output)
    print(json.dumps(res, indent=2))