import math
import argparse
import os
import cv2


def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    return img


def estimate_focal_length_pixels(focal_length_mm, sensor_width_mm, image_width_px):
    """Convert a physical focal length in mm to an approximate pixel focal length."""
    if focal_length_mm is None:
        raise ValueError("A focal length in mm is required when using physical focal length input.")
    if sensor_width_mm <= 0 or image_width_px <= 0:
        raise ValueError("Sensor width and image width must be positive.")
    return (focal_length_mm / sensor_width_mm) * image_width_px


def parse_pixel_coords(coord_string):
    """Parse a compact coordinate string like '314,249 311,182 305,245 323,244'."""
    values = coord_string.replace("\n", " ").split()
    if len(values) != 4:
        raise ValueError("Expected 4 coordinate pairs: 'x1,y1 x2,y2 x3,y3 x4,y4'.")

    parsed = []
    for item in values:
        try:
            x_str, y_str = item.split(",")
            parsed.append((float(x_str), float(y_str)))
        except ValueError as exc:
            raise ValueError(f"Invalid coordinate pair '{item}'.") from exc

    return parsed


def get_ground_distance(pixel_row, camera_height, tilt_deg, focal_px, cy):
    """Estimate the ground-plane distance to a point from the camera."""
    theta = math.radians(tilt_deg)
    alpha = theta + math.atan2(pixel_row - cy, focal_px)
    if alpha <= 0:
        return float("inf")
    return camera_height / math.tan(alpha)


def measure_object(base_row, top_row, left_col, right_col, camera_height, tilt_deg, focal_px, cy):
    """Estimate the real-world height and width of a vertical object from image geometry."""
    if focal_px <= 0:
        raise ValueError("Focal length must be positive.")

    dist_base = get_ground_distance(base_row, camera_height, tilt_deg, focal_px, cy)
    if math.isinf(dist_base):
        raise ValueError("The base point is outside the visible ground plane for the given camera pose.")

    theta = math.radians(tilt_deg)
    alpha_base = theta + math.atan2(base_row - cy, focal_px)
    alpha_top = theta + math.atan2(top_row - cy, focal_px)

    # For a vertical object standing on the same ground point, the height is inferred from
    # the difference between the ground-plane ray angle and the ray angle to the top.
    height_real = camera_height * (1.0 - (math.tan(alpha_top) / math.tan(alpha_base)))
    if height_real < 0:
        height_real = 0.0

    pixel_width = abs(right_col - left_col)
    width_real = dist_base * (pixel_width / focal_px)

    return dist_base, height_real, width_real


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A5 CCTV Object Measurement")
    parser.add_argument("--base-row", type=float, default=None, help="Y-coord of the object base")
    parser.add_argument("--top-row", type=float, default=None, help="Y-coord of the object top")
    parser.add_argument("--left-col", type=float, default=None, help="X-coord of the left edge")
    parser.add_argument("--right-col", type=float, default=None, help="X-coord of the right edge")
    parser.add_argument("--pixel-coords", default=None, help="Four coordinate pairs in the form 'x1,y1 x2,y2 x3,y3 x4,y4'")
    parser.add_argument("--height", type=float, default=3.0, help="Camera height above the floor (meters)")
    parser.add_argument("--tilt", type=float, required=True, help="Tilt angle (degrees)")
    parser.add_argument("--focal", type=float, default=None, help="Focal length in pixels")
    parser.add_argument("--focal-mm", type=float, default=None, help="Physical focal length in mm")
    parser.add_argument("--sensor-width-mm", type=float, default=4.0, help="Camera sensor width in mm")
    parser.add_argument("--image-width-px", type=int, default=640, help="Image width in pixels")
    parser.add_argument("--cy", type=float, default=None, help="Principal point Y (defaults to image height / 2)")
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", default="a5_measurement_result.jpg")

    args = parser.parse_args()

    try:
        if args.pixel_coords:
            coords = parse_pixel_coords(args.pixel_coords)
            base_x, base_y = coords[0]
            top_x, top_y = coords[1]
            left_x, left_y = coords[2]
            right_x, right_y = coords[3]
            base_row = base_y
            top_row = top_y
            left_col = left_x
            right_col = right_x
        else:
            if None in (args.base_row, args.top_row, args.left_col, args.right_col):
                raise ValueError("Provide either --pixel-coords or all four coordinate arguments.")
            base_row = args.base_row
            top_row = args.top_row
            left_col = args.left_col
            right_col = args.right_col

        if args.cy is None:
            img = load_image(args.image)
            args.cy = img.shape[0] / 2.0

        focal_px = args.focal
        if focal_px is None:
            focal_px = estimate_focal_length_pixels(args.focal_mm, args.sensor_width_mm, args.image_width_px)

        dist, h_real, w_real = measure_object(
            base_row, top_row, left_col, right_col,
            args.height, args.tilt, focal_px, args.cy
        )

        img = load_image(args.image)
        cv2.line(img, (int(left_col), int(base_row)), (int(right_col), int(base_row)), (0, 255, 255), 2)
        mid_x = int((left_col + right_col) / 2)
        cv2.line(img, (mid_x, int(base_row)), (mid_x, int(top_row)), (0, 255, 0), 2)

        label = f"H:{h_real:.2f}m W:{w_real:.2f}m Dist:{dist:.2f}m"
        cv2.putText(img, label, (int(left_col), int(top_row) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        output_dir = os.path.dirname(args.output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        success = cv2.imwrite(args.output, img)
        if not success:
            raise IOError(f"Failed to write image to '{args.output}'.")

        print(f"Focal length used: {focal_px:.2f} px")
        print(f"Results: Dist={dist:.2f}m, Height={h_real:.2f}m, Width={w_real:.2f}m")
        print(f"Saved to {args.output}")

    except Exception as e:
        print(f"Error: {e}")