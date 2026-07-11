"""
A1 - Scale Reference Method
============================
Math:
    Place ONE object of known real-world size (a "reference", e.g. a coin,
    a credit card, an A4 sheet) in the same image plane as the object(s)
    you want to measure.

        pixels_per_metric = reference_width_px / reference_width_real

    Then for any other object in the SAME plane (same distance from camera,
    roughly co-planar with the reference):

        object_real_size = object_width_px / pixels_per_metric

This method needs NO camera calibration and NO known focal length — it only
assumes the reference and target objects are roughly in the same plane and
photographed from the same viewpoint.
"""

import json
import argparse
from utils.camera_utils import (
    load_image, find_contours, min_area_rect_dimensions,
    draw_rotated_box, save_image, COLOR_REFERENCE, COLOR_TARGET,
)


def measure_objects(image_path, reference_width_real, reference_index=0, output_path=None):
    """
    Calculates sizes of all objects in the image based on a known reference.
    """
    img = load_image(image_path)
    contours = find_contours(img)

    if len(contours) < 2:
        raise ValueError(
            "Need at least 2 objects in frame: 1 reference + 1+ targets. "
            f"Found {len(contours)}."
        )

    # Use the Euclidean Distance logic inside min_area_rect_dimensions
    ref_box, ref_width_px, _ = min_area_rect_dimensions(contours[reference_index])
    pixels_per_metric = ref_width_px / reference_width_real

    annotated = img.copy() if output_path else None
    if annotated is not None:
        draw_rotated_box(
            annotated, ref_box, color=COLOR_REFERENCE,
            label=f"REF {reference_width_real:.2f}",
        )

    results = []
    for i, c in enumerate(contours):
        if i == reference_index:
            continue
        box, width_px, height_px = min_area_rect_dimensions(c)
        real_width = width_px / pixels_per_metric
        real_height = height_px / pixels_per_metric
        
        results.append({
            "width": real_width,
            "height": real_height,
            "box": box.tolist(),
        })
        
        if annotated is not None:
            draw_rotated_box(
                annotated, box, color=COLOR_TARGET,
                label=f"{real_width:.2f} x {real_height:.2f}",
            )

    output = {"pixels_per_metric": pixels_per_metric, "objects": results}

    if annotated is not None:
        save_image(annotated, output_path)
        output["annotated_image"] = output_path

    return output


if __name__ == "__main__":
    # Make A1 directly executable from the terminal!
    parser = argparse.ArgumentParser(description="A1 - Scale Reference Method")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--ref-width", type=float, required=True, help="Real-world width of reference object")
    parser.add_argument("--ref-index", type=int, default=0, help="Contour index of reference object (left-to-right)")
    parser.add_argument("--output", default="a1_output_annotated.jpg", help="Path to save visual output")
    
    args = parser.parse_args()

    print("Running A1: Scale Reference Method...")
    try:
        final_result = measure_objects(args.image, args.ref_width, args.ref_index, args.output)
        print(json.dumps(final_result, indent=2))
        print(f"\nSuccess! Annotated image saved to: {final_result['annotated_image']}")
    except Exception as e:
        print(f"\nError: {e}")