"""
A5 - Coordinate Geometry Method (Ground-Plane Ray Intersection)
==================================================================
Math (pinhole camera + known mounting geometry):

    Given:
        h        = camera height above the ground plane (meters)
        theta    = camera tilt/depression angle below horizontal (degrees)
        f        = focal length in pixels
        cy       = vertical pixel coordinate of the principal point (image center)
        v        = pixel row (y-coordinate) of the object's base/contact point

    The viewing ray angle (alpha) below horizontal is:
        alpha = theta + atan((v - cy) / f)

    Distance along ground (d):
        d = h / tan(alpha)

This method requires knowing your camera's height and tilt angle. It is
incredibly robust for CCTV/security setups looking at a flat floor.
"""

import math
import argparse
from utils.camera_utils import load_image, save_image, draw_horizontal_marker

def ground_distance_from_pixel(pixel_row, camera_height, tilt_angle_deg,
                                focal_length_px, principal_point_y,
                                image_path=None, output_path=None):
    """
    Computes ground-plane distance to an object's base.
    """
    theta = math.radians(tilt_angle_deg)
    # OpenCV: Y increases downwards. 
    # (v - cy) is positive for pixels below the horizon.
    pixel_offset = pixel_row - principal_point_y

    alpha = theta + math.atan(pixel_offset / focal_length_px)

    if alpha <= 0:
        raise ValueError("Object is above the horizon line; ground-plane assumption invalid.")

    distance = camera_height / math.tan(alpha)

    if image_path and output_path:
        img = load_image(image_path)
        draw_horizontal_marker(img, pixel_row, label=f"Dist: {distance:.2f}m")
        save_image(img, output_path)

    return distance

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A5 - CCTV Ground Plane Distance")
    parser.add_argument("--row", type=float, required=True, help="Pixel row of object base")
    parser.add_argument("--height", type=float, required=True, help="Camera height (meters)")
    parser.add_argument("--tilt", type=float, required=True, help="Tilt angle (degrees)")
    parser.add_argument("--focal", type=float, required=True, help="Focal length (pixels)")
    parser.add_argument("--cy", type=float, required=True, help="Principal point Y (usually image_height / 2)")
    parser.add_argument("--image", help="Optional image to annotate")
    parser.add_argument("--output", default="a5_result.jpg", help="Output path")
    
    args = parser.parse_args()

    try:
        dist = ground_distance_from_pixel(
            args.row, args.height, args.tilt, args.focal, args.cy,
            args.image, args.output
        )
        print(f"Calculated distance: {dist:.2f} meters")
    except Exception as e:
        print(f"Error: {e}")