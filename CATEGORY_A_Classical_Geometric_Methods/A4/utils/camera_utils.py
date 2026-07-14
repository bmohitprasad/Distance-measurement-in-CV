import os
import cv2
import numpy as np

# Color definitions for consistency across visualizations
COLOR_REFERENCE = (0, 140, 255)   # Orange (BGR)
COLOR_TARGET = (0, 220, 0)        # Green (BGR)
COLOR_LINE = (0, 255, 255)        # Yellow (BGR)
COLOR_POINT = (0, 0, 255)         # Red (BGR)
COLOR_TEXT = (255, 255, 255)      # White

def load_image(image_path):
    """Loads an image from disk, raising an error if not found."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    return img

def save_image(img, output_path):
    """Saves an image, automatically creating parent directories if needed."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path

def put_label(img, text, org, color=COLOR_TEXT, font_scale=0.6, thickness=2, bg=True):
    """Draws text with a dark background box for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
    if bg:
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x - 2, y - th - baseline - 4), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img

def draw_point(img, point, color=COLOR_POINT, radius=6, label=None, label_color=COLOR_TEXT):
    """Draws a highly visible circular marker at a specific pixel."""
    x, y = int(point[0]), int(point[1])
    cv2.circle(img, (x, y), radius, color, -1)
    cv2.circle(img, (x, y), radius + 2, (0, 0, 0), 1)
    if label:
        put_label(img, label, (x + radius + 4, y - radius - 4), color=label_color)
    return img

def draw_line(img, pt_a, pt_b, color=COLOR_LINE, thickness=2, label=None, label_color=COLOR_TEXT, label_offset=22):
    """Draws a connecting line between two points, used to visualize distance."""
    pa = (int(pt_a[0]), int(pt_a[1]))
    pb = (int(pt_b[0]), int(pt_b[1]))
    cv2.line(img, pa, pb, color, thickness)
    if label:
        # Place label near the midpoint of the line
        mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2 + label_offset)
        put_label(img, label, mid, color=label_color)
    return img