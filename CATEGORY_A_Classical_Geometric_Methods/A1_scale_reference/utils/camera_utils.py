"""
camera_utils.py
----------------
Shared helper functions used by the classical geometric methods.

Core idea used repeatedly across methods:
    pixels_per_metric = pixels_of_known_object / real_size_of_known_object

Once you have that ratio, any other pixel measurement in the SAME image
plane can be converted to real-world units:
    real_size = pixels_measured / pixels_per_metric
"""

import os
import cv2
import numpy as np


def load_image(image_path):
    """Load an image from disk and raise a clear error if it's missing/unreadable."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(
            f"Could not read image at '{image_path}'. "
            "Check the path and that the file is a valid image."
        )
    return img


def to_gray_blur(img, ksize=(5, 5)):
    """Grayscale + Gaussian blur — standard pre-processing before edge/contour detection."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, ksize, 0)


def find_contours(img, canny_low=50, canny_high=150, dilate_iter=1, max_area_frac=0.97):
    """
    Standard contour-extraction pipeline:
    blur -> Canny edges -> dilate (close small gaps) -> find contours.

    Uses cv2.RETR_LIST so nested objects are not silently dropped.
    Returns contours sorted left-to-right.
    """
    gray = to_gray_blur(img)
    edged = cv2.Canny(gray, canny_low, canny_high)
    edged = cv2.dilate(edged, None, iterations=dilate_iter)
    edged = cv2.erode(edged, None, iterations=dilate_iter)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    img_area = img.shape[0] * img.shape[1]
    contours = [
        c for c in contours
        if cv2.contourArea(c) > 100 and cv2.contourArea(c) < img_area * max_area_frac
    ]

    if not contours:
        return []

    contours = _deduplicate_by_bbox_overlap(contours)

    if not contours:
        return []

    # Sort left-to-right by the x-coordinate of each contour's bounding box
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    contours, _ = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][0]))
    return list(contours)


def _bbox_iou(box_a, box_b):
    """Intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1, aw, ah = box_a
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx1, by1, bw, bh = box_b
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_by_bbox_overlap(contours, iou_thresh=0.85):
    """
    Collapses near-duplicate bounding boxes down to just the largest contour
    among each overlapping group, solving the RETR_LIST double-contour issue.
    """
    boxes = [cv2.boundingRect(c) for c in contours]
    areas = [cv2.contourArea(c) for c in contours]
    order = sorted(range(len(contours)), key=lambda i: areas[i], reverse=True)

    kept = []
    kept_boxes = []
    for i in order:
        if any(_bbox_iou(boxes[i], kb) > iou_thresh for kb in kept_boxes):
            continue
        kept.append(contours[i])
        kept_boxes.append(boxes[i])
    return kept


def min_area_rect_dimensions(contour):
    """
    Returns (box_points, width_px, height_px) for the minimum-area rotated
    bounding rectangle of a contour. Uses Euclidean distance internally.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype="float64")

    (tl, tr, br, bl) = order_points(box)
    width_px = euclidean(tl, tr)
    height_px = euclidean(tl, bl)
    return box, width_px, height_px


def order_points(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float64")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   
    rect[2] = pts[np.argmax(s)]   

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] 
    rect[3] = pts[np.argmax(diff)] 
    return rect


def euclidean(ptA, ptB):
    """Calculates Euclidean distance between two pixel coordinates."""
    return float(np.sqrt((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
COLOR_REFERENCE = (0, 140, 255)   # Orange (BGR)
COLOR_TARGET = (0, 220, 0)        # Green (BGR)
COLOR_TEXT = (255, 255, 255)      # White


def put_label(img, text, org, color=COLOR_TEXT, font_scale=0.6, thickness=2, bg=True):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
    if bg:
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(img, (x - 2, y - th - baseline - 4), (x + tw + 4, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return img


def draw_rotated_box(img, box, color=COLOR_TARGET, thickness=2, label=None, label_color=COLOR_TEXT):
    pts = np.array(box, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
    if label:
        top_idx = int(np.argmin(pts[:, 1]))
        top_pt = pts[top_idx]
        put_label(img, label, (top_pt[0], max(20, top_pt[1] - 10)), color=label_color)
    return img


def save_image(img, output_path):
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path