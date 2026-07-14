"""
A2 - Single-Image Photogrammetry (Triangle Similarity Method)
============================================================================
Math (similar triangles):

    Calibration step (done once, offline):
        F = (P_known x D_known) / W_real

    Measurement step (any new photo, same camera/lens/zoom):
        D = (W_real x F) / P_measured

This script utilizes a sub-command CLI to separate the `calibrate` step
from the `measure` step.
"""

import json
import argparse
from utils.camera_utils import (
    load_image, find_contours_all, select_target_contour,
    min_area_rect_dimensions, focal_length_from_known_distance,
    distance_from_focal_length, draw_rotated_box, save_image,
    diagnose_contours, COLOR_REFERENCE, COLOR_TARGET
)


def calibrate_focal_length(image_path, known_distance, known_width_real, output_path=None, debug=False):
    img = load_image(image_path)

    if debug:
        print("--- Calibration Image Diagnostics ---")
        diagnose_contours(img)

    contours = find_contours_all(img)
    target = select_target_contour(contours, img.shape)
    
    if target is None:
        raise ValueError(
            "No valid target contour found in calibration image. "
            "Ensure the object is clearly visible and does not touch the edges."
        )

    box, width_px, _ = min_area_rect_dimensions(target)
    focal_length = focal_length_from_known_distance(known_distance, known_width_real, width_px)

    if output_path:
        annotated = img.copy()
        draw_rotated_box(
            annotated, box, color=COLOR_REFERENCE,
            label=f"CALIB D={known_distance:.2f} W={known_width_real:.2f} P={width_px:.1f}px",
        )
        save_image(annotated, output_path)

    return {"focal_length_px": focal_length, "annotated_image": output_path}


def estimate_distance(image_path, focal_length, known_width_real, output_path=None, debug=False):
    img = load_image(image_path)

    if debug:
        print("--- Measurement Image Diagnostics ---")
        diagnose_contours(img)

    contours = find_contours_all(img)
    target = select_target_contour(contours, img.shape)
    
    if target is None:
        raise ValueError(
            "No valid target contour found in measurement image. "
            "Ensure the object is clearly visible and does not touch the edges."
        )

    box, width_px, _ = min_area_rect_dimensions(target)
    distance = distance_from_focal_length(focal_length, known_width_real, width_px)

    if output_path:
        annotated = img.copy()
        draw_rotated_box(
            annotated, box, color=COLOR_TARGET,
            label=f"D = {distance:.2f} P={width_px:.1f}px",
        )
        save_image(annotated, output_path)

    return {"estimated_distance": distance, "annotated_image": output_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2 - Single-Image Photogrammetry")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode to run: 'calibrate' or 'measure'")

    # Subparser for Calibration
    calib_parser = subparsers.add_parser("calibrate", help="Calibrate camera focal length")
    calib_parser.add_argument("--image", required=True, help="Path to calibration image")
    calib_parser.add_argument("--distance", type=float, required=True, help="Known distance to object")
    calib_parser.add_argument("--width", type=float, required=True, help="Known real-world width of object")
    calib_parser.add_argument("--output", default="a2_calib_annotated.jpg", help="Path to save annotated image")
    calib_parser.add_argument("--debug", action="store_true", help="Print contour diagnostic information")

    # Subparser for Measurement
    meas_parser = subparsers.add_parser("measure", help="Measure distance in a new photo")
    meas_parser.add_argument("--image", required=True, help="Path to measurement image")
    meas_parser.add_argument("--focal", type=float, required=True, help="Focal length obtained from calibration")
    meas_parser.add_argument("--width", type=float, required=True, help="Known real-world width of object")
    meas_parser.add_argument("--output", default="a2_measure_annotated.jpg", help="Path to save annotated image")
    meas_parser.add_argument("--debug", action="store_true", help="Print contour diagnostic information")

    args = parser.parse_args()

    try:
        if args.mode == "calibrate":
            print(f"Running A2 Calibration on {args.image}...")
            result = calibrate_focal_length(args.image, args.distance, args.width, args.output, args.debug)
            print(json.dumps(result, indent=2))
        
        elif args.mode == "measure":
            print(f"Running A2 Measurement on {args.image}...")
            result = estimate_distance(args.image, args.focal, args.width, args.output, args.debug)
            print(json.dumps(result, indent=2))
            
    except Exception as e:
        print(f"\nError: {e}")