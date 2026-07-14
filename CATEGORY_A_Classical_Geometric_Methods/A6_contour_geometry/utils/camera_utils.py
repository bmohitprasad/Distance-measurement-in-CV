"""
camera_utils.py
----------------
Helper functions for Contour Geometry. 
Includes standardized contour extraction and visualization logic.
"""

import os
import cv2
import numpy as np

COLOR_REFERENCE = (0, 140, 255)   # Orange (BGR)
COLOR_TARGET = (0, 220, 0)        # Green (BGR)
COLOR_TEXT = (255, 255, 255)      # White

def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    return img

def to_gray_blur(img, ksize=(5, 5)):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, ksize, 0)

def find_contours(img, canny_low=50, canny_high=150):
    """Pipeline to extract contours for irregular shapes."""
    gray = to_gray_blur(img)
    edged = cv2.Canny(gray, canny_low, canny_high)
    # Morphological operations to close small gaps in contours
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def min_area_rect_dimensions(contour):
    """Returns (box, width, height) of the rotated bounding box."""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect)
    return box, max(w, h), min(w, h)

def draw_contour(img, contour, color=COLOR_TARGET, thickness=2, label=None):
    """Draws the contour outline and an optional label."""
    cv2.drawContours(img, [contour], -1, color, thickness)
    if label:
        x, y, w, h = cv2.boundingRect(contour)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, 0.5, 1)
        cv2.rectangle(img, (x, y - th - 5), (x + tw, y), (0, 0, 0), -1)
        cv2.putText(img, label, (x, y - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img

def save_image(img, output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, img)