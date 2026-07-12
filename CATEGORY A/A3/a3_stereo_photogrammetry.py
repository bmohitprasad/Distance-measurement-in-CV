"""
A3 - Stereo Photogrammetry
============================
Math (stereo triangulation / disparity):

    depth = (focal_length x baseline) / disparity

    where:
        focal_length = camera focal length in PIXELS (same units as image)
        baseline     = physical distance between the two camera centers
        disparity    = (x_left - x_right), the horizontal pixel shift of the
                       SAME physical point between the left and right images

Close objects have LARGE disparity (they shift a lot between the two
views); far objects have SMALL disparity. This is exactly how human/animal
binocular vision perceives depth.

Pipeline:
    1. Capture a synchronized stereo pair (two cameras, known baseline).
    2. Compute a disparity map (block matching / semi-global matching).
    3. Convert disparity -> depth using the formula above, per pixel.
"""

import cv2
import numpy as np
import argparse
import os

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    return img

def compute_disparity_map(left_image_path, right_image_path, num_disparities=16 * 6, block_size=15):
    """
    Compute a disparity map from a rectified stereo pair using OpenCV's
    block-matching stereo algorithm (StereoBM).
    """
    left = load_image(left_image_path)
    right = load_image(right_image_path)

    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

    return disparity

def disparity_to_depth_map(disparity_map, focal_length_px, baseline, min_disparity=0.5):
    """
    Convert a raw disparity map into a metric depth map.
    Pixels with disparity below `min_disparity` are treated as invalid
    (would imply infinite / unreliable depth) and set to NaN.
    """
    safe_disparity = np.where(disparity_map > min_disparity, disparity_map, np.nan)
    depth_map = (focal_length_px * baseline) / safe_disparity
    return depth_map

def depth_at_point(depth_map, x, y):
    """Read the estimated depth at a single pixel coordinate (x, y)."""
    return float(depth_map[y, x])

def visualize_depth_map(depth_map, output_path, base_image_path=None):
    """
    Render the depth map as a color heatmap and save it.
    (Thermal camera style: near = warm colors, far = cool colors)
    """
    valid = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0)
    finite_vals = valid[valid > 0]
    if finite_vals.size == 0:
        raise ValueError("Depth map has no valid (finite, positive) values to visualize.")

    d_min, d_max = float(finite_vals.min()), float(finite_vals.max())
    # Normalize inverted so closer = warmer colors (red), farther = cooler colors (blue)
    normalized = 1.0 - np.clip((valid - d_min) / max(d_max - d_min, 1e-6), 0, 1)
    normalized_u8 = (normalized * 255).astype(np.uint8)
    
    heatmap = cv2.applyColorMap(normalized_u8, cv2.COLORMAP_JET)
    heatmap[valid == 0] = (0, 0, 0)  # mark invalid pixels black

    if base_image_path:
        base = load_image(base_image_path)
        base = cv2.resize(base, (heatmap.shape[1], heatmap.shape[0]))
        canvas = cv2.addWeighted(base, 0.45, heatmap, 0.55, 0)
    else:
        canvas = heatmap

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, canvas)
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A3 - Stereo Photogrammetry Depth Map")
    parser.add_argument("--left", required=True, help="Path to Left image")
    parser.add_argument("--right", required=True, help="Path to Right image")
    parser.add_argument("--focal", type=float, default=718.856, help="Focal length in pixels (KITTI default: 718.856)")
    parser.add_argument("--baseline", type=float, default=0.54, help="Baseline distance in meters (KITTI default: 0.54)")
    parser.add_argument("--output", default="a3_depth_heatmap.jpg", help="Path to save visual heatmap")
    
    args = parser.parse_args()

    print("Computing Disparity Map...")
    disparity = compute_disparity_map(args.left, args.right)
    
    print("Converting to Depth Map...")
    depth_map = disparity_to_depth_map(disparity, focal_length_px=args.focal, baseline=args.baseline)
    
    h, w = depth_map.shape
    center_depth = depth_at_point(depth_map, w // 2, h // 2)
    print(f"Success! Estimated depth at center pixel: {center_depth:.2f} meters")
    
    visualize_depth_map(depth_map, args.output, base_image_path=args.left)
    print(f"Depth heatmap saved to: {args.output}")