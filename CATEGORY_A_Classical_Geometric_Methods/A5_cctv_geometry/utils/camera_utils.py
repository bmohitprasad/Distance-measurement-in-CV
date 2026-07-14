"""
camera_utils.py
----------------
Helper functions for A5 CCTV Ground-Plane Geometry.
"""

import os
import cv2

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    return img

def save_image(img, output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path

def draw_horizontal_marker(img, pixel_row, label=None, color=(0, 255, 255), thickness=2):
    """Draws a horizontal line across the image at a specific row."""
    h, w = img.shape[:2]
    row = int(pixel_row)
    cv2.line(img, (0, row), (w, row), color, thickness)
    if label:
        cv2.putText(img, label, (10, max(20, row - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img