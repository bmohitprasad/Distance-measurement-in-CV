"""
camera_utils.py
----------------
Shared helper functions tailored for A2 Single-Image Photogrammetry.

This includes the robust contour selection logic (filtering out backgrounds
and selecting the largest valid object) required for accurate A2 measurement.
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


def find_contours_all(img, canny_low=50, canny_high=150, dilate_iter=1):
    """Finds all contours using RETR_LIST."""
    gray = to_gray_blur(img)
    edged = cv2.Canny(gray, canny_low, canny_high)
    edged = cv2.dilate(edged, None, iterations=dilate_iter)
    edged = cv2.erode(edged, None, iterations=dilate_iter)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def select_target_contour(contours, img_shape, max_area_frac=0.97):
    """
    Filters out background noise and full-frame artifacts, then selects 
    the largest remaining contour as the target object.
    """
    img_area = img_shape[0] * img_shape[1]
    valid_contours = []
    
    for c in contours:
        area = cv2.contourArea(c)
        if area < 100:
            continue
        if area > img_area * max_area_frac:
            continue
            
        x, y, w, h = cv2.boundingRect(c)
        borders_touched = 0
        if x <= 5: borders_touched += 1
        if y <= 5: borders_touched += 1
        if (x + w) >= img_shape[1] - 5: borders_touched += 1
        if (y + h) >= img_shape[0] - 5: borders_touched += 1
        
        if borders_touched > 1:
            continue
            
        valid_contours.append(c)

    if not valid_contours:
        return None

    # Deduplicate overlapping bounding boxes (Canny often draws double lines)
    valid_contours = _deduplicate_by_bbox_overlap(valid_contours)
    
    if not valid_contours:
        return None

    # Return the largest valid contour
    return max(valid_contours, key=cv2.contourArea)


def _bbox_iou(box_a, box_b):
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
    Returns (box_points, width_px, height_px).
    Ensures 'width' is always the longer side for consistency.
    """
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect)
    
    width_px = max(w, h)
    height_px = min(w, h)
    return box, width_px, height_px


def focal_length_from_known_distance(known_distance, known_width_real, known_width_px):
    return (known_width_px * known_distance) / known_width_real


def distance_from_focal_length(focal_length, known_width_real, perceived_width_px):
    return (known_width_real * focal_length) / perceived_width_px


def diagnose_contours(img):
    """Prints terminal diagnostics for contour selection debugging."""
    contours = find_contours_all(img)
    img_area = img.shape[0] * img.shape[1]
    
    print(f"\n[DEBUG] Found {len(contours)} total raw contours.")
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        borders = sum([x <= 5, y <= 5, (x+w) >= img.shape[1]-5, (y+h) >= img.shape[0]-5])
        frac = (area / img_area) * 100
        print(f"  #{i}: Area={area:.0f} ({frac:.1f}% of frame), Borders touched={borders}")
    print("")


def put_label(img, text, org, color=COLOR_TEXT, font_scale=0.6, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    x, y = int(org[0]), int(org[1])
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