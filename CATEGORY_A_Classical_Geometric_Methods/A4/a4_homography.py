"""
A4 - Homography Method
========================
Math:
    A homography H is a 3x3 matrix mapping points on a flat plane in the
    image (pixel coords) to real-world coordinates on that same plane:

        [X_world]       [u_pixel]
        [Y_world] = H * [v_pixel]
        [   1   ]       [   1   ]

    H is solved from >= 4 point correspondences using cv2.findHomography.
    Once calculated, any pixel on the targeted plane (e.g., the ground)
    can be projected into accurate physical coordinates.
"""

import cv2
import numpy as np
import argparse
import json

from utils.camera_utils import (
    load_image, save_image, draw_line, draw_point,
    COLOR_REFERENCE, COLOR_POINT,
)

def compute_homography(image_points, world_points):
    """
    Calculates the 3x3 Homography matrix mapping pixels to real-world coordinates.
    Requires at least 4 corresponding points.
    """
    image_points = np.array(image_points, dtype="float32")
    world_points = np.array(world_points, dtype="float32")

    # Use RANSAC to be robust against slight clicking errors
    H, mask = cv2.findHomography(image_points, world_points, method=cv2.RANSAC)
    if H is None:
        raise ValueError("Homography computation failed. Ensure you have >= 4 valid points.")
    return H

def pixel_to_world(H, pixel_point):
    """
    Applies the Homography matrix to a single (x, y) pixel coordinate to
    extract its real-world (X, Y) coordinate on the calibrated plane.
    """
    # Create homogeneous coordinate [x, y, 1]
    px = np.array([pixel_point[0], pixel_point[1], 1.0], dtype="float64")
    
    # Multiply by the matrix
    world = H @ px
    
    # De-homogenize (divide by Z to return to 2D)
    world = world / world[2]
    return float(world[0]), float(world[1])

def real_world_distance(H, pixel_point_a, pixel_point_b, image_path=None, output_path=None):
    """
    Calculates the real-world distance between two arbitrary pixels on the calibrated plane.
    """
    xa, ya = pixel_to_world(H, pixel_point_a)
    xb, yb = pixel_to_world(H, pixel_point_b)
    
    # Standard Euclidean distance on the real-world coordinates
    distance = float(np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2))

    if image_path and output_path:
        img = load_image(image_path)
        draw_line(img, pixel_point_a, pixel_point_b, label=f"{distance:.2f} units")
        draw_point(img, pixel_point_a, color=COLOR_POINT, label="A")
        draw_point(img, pixel_point_b, color=COLOR_POINT, label="B")
        save_image(img, output_path)

    return distance

def parse_points(point_string):
    """Converts a string like '120,400 520,400' into a list of tuples [(120.0, 400.0), ...]."""
    points = []
    for pair in point_string.strip().split():
        x, y = pair.split(',')
        points.append((float(x), float(y)))
    return points

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A4 - Homography Measurement")
    parser.add_argument("--image", required=True, help="Path to the image")
    parser.add_argument("--img-pts", required=True, help="4+ image pixel coordinates, space separated pairs (e.g. '120,400 520,400 520,100 120,100')")
    parser.add_argument("--world-pts", required=True, help="4+ corresponding real-world coordinates (e.g. '0,0 2,0 2,1 0,1')")
    parser.add_argument("--measure-pts", required=True, help="2 pixel coordinates to measure the distance between (e.g. '200,350 450,150')")
    parser.add_argument("--output", default="a4_annotated.jpg", help="Path to save annotated image")
    
    args = parser.parse_args()

    try:
        print("Parsing coordinates...")
        image_pts = parse_points(args.img_pts)
        world_pts = parse_points(args.world_pts)
        measure_pts = parse_points(args.measure_pts)

        if len(image_pts) < 4 or len(world_pts) < 4:
            raise ValueError("You must provide at least 4 image points and 4 world points.")
        if len(measure_pts) != 2:
            raise ValueError("You must provide exactly 2 points to measure the distance between.")

        print("Computing Homography Matrix (H)...")
        H = compute_homography(image_pts, world_pts)
        
        print("Calculating real-world distance...")
        distance = real_world_distance(H, measure_pts[0], measure_pts[1], 
                                       image_path=args.image, output_path=args.output)
        
        result = {
            "homography_matrix": H.tolist(),
            "measured_distance": distance,
            "output_image": args.output
        }
        
        print(json.dumps(result, indent=2))
        print(f"\nSuccess! Annotated image saved to: {args.output}")

    except Exception as e:
        print(f"\nError: {e}")