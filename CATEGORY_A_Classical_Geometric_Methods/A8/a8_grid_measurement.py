"""
METHOD 8: Grid Reference Method
==================================
WHAT IT DOES:
    A regular grid pattern (graph paper, printed grid, floor tiles) appears
    in the image. You count how many grid squares span the object. Multiply
    by the known grid square size to get real dimensions. No camera
    calibration required.

WHAT IMAGE TO USE:
    Best option — print this and use it as background:
        A4 graph paper with 5mm or 1cm grid squares

    Other good options:
        - Notebook/exercise book grid paper
        - Printed checkerboard pattern (can generate below)
        - Floor tiles of known size (30×30cm bathroom tiles)
        - Spreadsheet printed as grid

    Setup:
        Place object ON TOP of graph paper.
        Photograph from directly above (top-down view).
        Keep camera parallel to paper surface — avoid tilt.
        Ensure grid lines are visible around the object.

EXAMPLE IMAGES:
    - Rice grain on 1mm graph paper (measuring ~7mm length)
    - Mango on 1cm graph paper (measuring ~12cm height)
    - Coin on grid paper (measuring diameter)
    - Leaf on grid paper (measuring span + estimating area)

HOW TO RUN:
    # Run with demo image (auto-generated grid + object):
    python method8_grid.py

    # Run with your own image:
    python method8_grid.py --image images/method8_grid/rice_on_graph.jpg

    # Specify grid and object parameters:
    python method8_grid.py --grid_cm 1.0 --grid_px 40

    # Interactive mode — click object corners on image:
    python method8_grid.py --image your_image.jpg --interactive

    # Auto-detect grid lines from image:
    python method8_grid.py --image your_image.jpg --auto_detect

    # Generate a printable grid image:
    python method8_grid.py --generate_grid --grid_cm 1.0

PARAMETERS:
    --grid_cm       : real size of one grid square in cm (e.g. 0.5, 1.0, 5.0)
    --grid_px       : pixel size of one grid square in image
                      (measure: count pixels between two adjacent grid lines)
    --obj_squares_w : object width in grid squares (can be fractional, e.g. 3.5)
    --obj_squares_h : object height in grid squares (can be fractional)
    --obj_x         : top-left X of object bounding box in pixels
    --obj_y         : top-left Y of object bounding box in pixels
    --interactive   : click top-left and bottom-right corners of object
    --auto_detect   : attempt to detect grid lines automatically
    --generate_grid : save a printable grid image to print_this_grid.png

HOW TO MEASURE grid_px FROM YOUR IMAGE:
    Open image → zoom in → count pixels between two consecutive grid lines
    In GIMP:    use the measure tool between two grid lines
    In MS Paint: hover over start of grid line (note X coord), hover over
                 next grid line (note X coord), subtract
    In VS Code:  install "Pixel Info" extension → hover over pixels

HOW TO MEASURE obj_squares_w FROM IMAGE:
    Count grid squares from left edge to right edge of object.
    Fractional squares are fine — estimate 0.5 for half a square.
    Example: object spans 3 full squares + about half = 3.5
"""

import cv2
import numpy as np
import argparse
import os
import sys
import math

# ── helpers ──────────────────────────────────────────────────────────────────

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Image not found at: {path}")
    return img

def save_result(img, suffix, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"method8_{suffix}.jpg")
    cv2.imwrite(path, img)
    print(f"  Saved → {path}")
    return path

def put_text(img, text, pos, color=(255,255,255), scale=0.55, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def show_metrics(metrics):
    print("\n" + "═"*50)
    for k, v in metrics.items():
        print(f"  {k:<32} {v}")
    print("═"*50 + "\n")

# ── grid generation ───────────────────────────────────────────────────────────

def generate_printable_grid(grid_cm=1.0, page_w_cm=21.0, page_h_cm=29.7,
                             dpi=96, out_path="print_this_grid.png"):
    """
    Generate a printable A4 grid image.
    Print at 100% scale (no fit-to-page) for accurate measurements.
    """
    px_per_cm = dpi / 2.54
    W = int(page_w_cm * px_per_cm)
    H = int(page_h_cm * px_per_cm)
    grid_px = int(grid_cm * px_per_cm)

    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255  # white background

    # Draw minor grid lines (thin, light gray)
    for x in range(0, W, grid_px):
        cv2.line(canvas, (x, 0), (x, H), (200, 200, 220), 1)
    for y in range(0, H, grid_px):
        cv2.line(canvas, (0, y), (W, y), (200, 200, 220), 1)

    # Draw major grid lines every 5 squares (thicker, darker)
    major = grid_px * 5
    for x in range(0, W, major):
        cv2.line(canvas, (x, 0), (x, H), (140, 140, 180), 1)
    for y in range(0, H, major):
        cv2.line(canvas, (0, y), (W, y), (140, 140, 180), 1)

    # Border
    cv2.rectangle(canvas, (2,2), (W-2,H-2), (80,80,120), 2)

    # Label
    label = f"Grid: {grid_cm} cm per square | Print at 100% scale, no scaling"
    cv2.putText(canvas, label, (10, H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,140), 1)

    cv2.imwrite(out_path, canvas)
    print(f"\n  Printable grid saved → {out_path}")
    print(f"  Grid square: {grid_cm} cm | Page: A4 ({page_w_cm}×{page_h_cm} cm)")
    print(f"  Print at EXACTLY 100% (no scaling) for accurate measurements.\n")
    return canvas, grid_px

# ── demo image generator ──────────────────────────────────────────────────────

def make_demo_image(grid_px=40, grid_cm=1.0, cols=16, rows=12,
                    out_path="images/method8_grid/demo.jpg"):
    """
    Synthetic image: grid background + object (rice-grain shaped ellipse).
    """
    W, H = cols * grid_px + 60, rows * grid_px + 60
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 248  # off-white

    # Grid lines
    for x in range(30, W-30, grid_px):
        cv2.line(canvas, (x, 30), (x, H-30), (180, 195, 220), 1)
    for y in range(30, H-30, grid_px):
        cv2.line(canvas, (30, y), (W-30, y), (180, 195, 220), 1)

    # Major lines every 5
    for x in range(30, W-30, grid_px*5):
        cv2.line(canvas, (x, 30), (x, H-30), (130, 150, 190), 1)
    for y in range(30, H-30, grid_px*5):
        cv2.line(canvas, (30, y), (W-30, y), (130, 150, 190), 1)

    # Border
    cv2.rectangle(canvas, (28,28), (W-28,H-28), (100,120,160), 2)

    # Grid labels (column numbers)
    for i, x in enumerate(range(30, W-30, grid_px)):
        if i % 5 == 0:
            cv2.putText(canvas, str(i), (x+2, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (130,130,170), 1)

    # Object: rice-grain shaped ellipse (~7 squares wide, 2.5 squares tall)
    obj_w_sq = 7.0
    obj_h_sq = 2.5
    obj_cx = 30 + int(cols/2 * grid_px)
    obj_cy = 30 + int(rows/2 * grid_px)
    axes = (int(obj_w_sq/2 * grid_px), int(obj_h_sq/2 * grid_px))

    # Draw object shadow
    shadow = canvas.copy()
    cv2.ellipse(shadow, (obj_cx+4, obj_cy+4), axes, 10, 0, 360, (180,180,180), -1)
    cv2.addWeighted(shadow, 0.4, canvas, 0.6, 0, canvas)

    # Draw object
    cv2.ellipse(canvas, (obj_cx, obj_cy), axes, 10, 0, 360, (140, 175, 110), -1)
    cv2.ellipse(canvas, (obj_cx, obj_cy), axes, 10, 0, 360, (80, 120, 60), 2)

    # Highlight
    hi_axes = (max(1,axes[0]-15), max(1,axes[1]-8))
    cv2.ellipse(canvas, (obj_cx-8, obj_cy-6), hi_axes, 10, 200, 310, (200,225,180), 2)

    # Caption
    cv2.putText(canvas,
        f"Demo: rice-like grain on {grid_cm}cm grid  |  object ~ {obj_w_sq} x {obj_h_sq} squares",
        (10, H-8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80,80,120), 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"  Demo image created → {out_path}")

    # Return object bounds for auto-fill
    ox = obj_cx - axes[0] - 5
    oy = obj_cy - axes[1] - 5
    ow = 2*axes[0] + 10
    oh = 2*axes[1] + 10
    return canvas, (ox, oy, ow, oh), obj_w_sq, obj_h_sq

# ── auto grid detection ───────────────────────────────────────────────────────

def detect_grid_spacing(image):
    """
    Attempt to detect grid line spacing using FFT on row/column projections.
    Returns estimated grid_px or None if detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Project columns
    col_proj = gray.mean(axis=0).astype(np.float32)
    col_proj -= col_proj.mean()

    # FFT to find dominant spatial frequency
    fft = np.abs(np.fft.rfft(col_proj))
    fft[0] = 0  # remove DC
    freqs = np.fft.rfftfreq(len(col_proj))

    # Find peak frequency (exclude very low and very high)
    min_f = 1/200  # grid squares at least 5px
    max_f = 1/5
    valid = (freqs > min_f) & (freqs < max_f)
    if not valid.any():
        return None, None

    peak_freq = freqs[valid][np.argmax(fft[valid])]
    grid_px = int(round(1.0 / peak_freq))

    print(f"  [Auto-detect] Estimated grid spacing: {grid_px} pixels")
    return grid_px, col_proj

# ── interactive point selection ───────────────────────────────────────────────

_clicked = []

def _mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicked) < 2:
        _clicked.append((x, y))
        label = "Top-left" if len(_clicked) == 1 else "Bottom-right"
        print(f"  {label} corner: ({x}, {y})")

def interactive_select(image):
    """Let user click top-left and bottom-right corners of object."""
    print("\n  Click TOP-LEFT corner of object, then BOTTOM-RIGHT corner.")
    print("  Press Q to confirm.\n")
    _clicked.clear()
    cv2.namedWindow("Select object corners — click 2 points, press Q")
    cv2.setMouseCallback("Select object corners — click 2 points, press Q", _mouse_cb)

    while True:
        disp = image.copy()
        for pt in _clicked:
            cv2.circle(disp, pt, 6, (0, 80, 220), -1)
        if len(_clicked) == 2:
            cv2.rectangle(disp, _clicked[0], _clicked[1], (0, 200, 80), 2)
        cv2.imshow("Select object corners — click 2 points, press Q", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or (key == 13 and len(_clicked) == 2):
            break

    cv2.destroyAllWindows()
    if len(_clicked) == 2:
        x1, y1 = _clicked[0]
        x2, y2 = _clicked[1]
        return min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)
    return None

# ── core measurement ──────────────────────────────────────────────────────────

def run(image, grid_cm, grid_px,
        obj_x, obj_y, obj_w_px, obj_h_px,
        obj_squares_w=None, obj_squares_h=None,
        show_overlay=True):
    """
    Core grid measurement function.

    Args:
        image        : BGR numpy array
        grid_cm      : real size of one grid square in cm
        grid_px      : pixel size of one grid square
        obj_x/y      : top-left pixel of object bounding box
        obj_w/h_px   : pixel dimensions of object bounding box
        obj_squares_w/h : override — if you counted squares manually
        show_overlay : draw grid overlay on output

    Returns:
        metrics dict with all measurements
    """

    vis = image.copy()
    h_img, w_img = image.shape[:2]

    scale = grid_cm / grid_px          # cm per pixel

    # Compute real dimensions
    if obj_squares_w is not None:
        real_w = obj_squares_w * grid_cm
        obj_w_px = obj_squares_w * grid_px
    else:
        obj_squares_w = obj_w_px / grid_px
        real_w = obj_squares_w * grid_cm

    if obj_squares_h is not None:
        real_h = obj_squares_h * grid_cm
        obj_h_px = obj_squares_h * grid_px
    else:
        obj_squares_h = obj_h_px / grid_px
        real_h = obj_squares_h * grid_cm

    real_area    = real_w * real_h
    diagonal_cm  = math.sqrt(real_w**2 + real_h**2)
    perimeter_cm = 2 * (real_w + real_h)

    # ── draw grid overlay ──────────────────────────────────────────────────
    if show_overlay:
        overlay = vis.copy()
        # Vertical grid lines
        for x in range(0, w_img, grid_px):
            cv2.line(overlay, (x, 0), (x, h_img), (100, 140, 220), 1)
        # Horizontal grid lines
        for y in range(0, h_img, grid_px):
            cv2.line(overlay, (0, y), (w_img, y), (100, 140, 220), 1)
        # Major lines every 5 squares
        for x in range(0, w_img, grid_px*5):
            cv2.line(overlay, (x, 0), (x, h_img), (60, 90, 200), 1)
        for y in range(0, h_img, grid_px*5):
            cv2.line(overlay, (0, y), (w_img, y), (60, 90, 200), 1)
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

    # ── draw object bounding box ───────────────────────────────────────────
    ox, oy = int(obj_x), int(obj_y)
    ow, oh = int(obj_w_px), int(obj_h_px)

    # Shade grid squares inside bounding box
    sq_shade = vis.copy()
    x_start = (ox // grid_px) * grid_px
    y_start = (oy // grid_px) * grid_px
    for gx in range(x_start, ox+ow, grid_px):
        for gy in range(y_start, oy+oh, grid_px):
            x1c = max(gx, 0);      y1c = max(gy, 0)
            x2c = min(gx+grid_px, w_img); y2c = min(gy+grid_px, h_img)
            cv2.rectangle(sq_shade, (x1c,y1c), (x2c,y2c), (60,200,120), -1)
    cv2.addWeighted(sq_shade, 0.18, vis, 0.82, 0, vis)

    # Bounding box
    cv2.rectangle(vis, (ox,oy), (ox+ow,oy+oh), (0,200,80), 2)

    # ── dimension arrows ───────────────────────────────────────────────────
    arrow_color = (0, 50, 220)
    arrow_y = oy - 18
    if arrow_y < 15:
        arrow_y = oy + oh + 22

    # Horizontal arrow (width)
    cv2.arrowedLine(vis, (ox, arrow_y), (ox+ow, arrow_y), arrow_color, 2, tipLength=0.08)
    cv2.arrowedLine(vis, (ox+ow, arrow_y), (ox, arrow_y), arrow_color, 2, tipLength=0.08)
    w_label = f"{real_w:.2f} cm  ({obj_squares_w:.1f} sq)"
    put_text(vis, w_label, (ox + ow//2 - len(w_label)*4, arrow_y - 6),
             (0,50,220), 0.5)

    # Vertical arrow (height)
    arrow_x = ox - 22
    if arrow_x < 15:
        arrow_x = ox + ow + 12
    cv2.arrowedLine(vis, (arrow_x, oy), (arrow_x, oy+oh), arrow_color, 2, tipLength=0.08)
    cv2.arrowedLine(vis, (arrow_x, oy+oh), (arrow_x, oy), arrow_color, 2, tipLength=0.08)
    h_label = f"{real_h:.2f}cm ({obj_squares_h:.1f}sq)"
    label_y = oy + oh//2
    for i, ch in enumerate(h_label):
        put_text(vis, ch, (arrow_x - 14, label_y - len(h_label)*6 + i*12),
                 (0,50,220), 0.38)

    # ── corner markers ─────────────────────────────────────────────────────
    for corner in [(ox,oy),(ox+ow,oy),(ox+ow,oy+oh),(ox,oy+oh)]:
        cv2.circle(vis, corner, 4, (0,80,220), -1)

    # ── info panel ────────────────────────────────────────────────────────
    panel_y = h_img - 52
    cv2.rectangle(vis, (0, panel_y), (w_img, h_img), (20,20,30), -1)
    put_text(vis, f"Grid: {grid_cm} cm/sq  |  1 sq = {grid_px} px  |  Scale: {scale:.4f} cm/px",
             (8, panel_y+16), (180,200,240), 0.42)
    put_text(vis,
             f"Width: {real_w:.3f} cm ({obj_squares_w:.2f} sq)   "
             f"Height: {real_h:.3f} cm ({obj_squares_h:.2f} sq)   "
             f"Area: {real_area:.3f} cm²",
             (8, panel_y+36), (100,240,160), 0.48)

    # ── grid square count visualization ───────────────────────────────────
    # Draw a separate panel showing fractional squares counted
    count_h = 120
    count_w = w_img
    count_canvas = np.ones((count_h, count_w, 3), dtype=np.uint8) * 28

    sq_viz_size = min(40, int((count_w - 40) / (obj_squares_w + 2)))
    sq_viz_size = max(sq_viz_size, 8)
    full_w = int(obj_squares_w)
    frac_w = obj_squares_w - full_w
    full_h = int(obj_squares_h)
    frac_h = obj_squares_h - full_h

    put_text(count_canvas, "Grid square count visualization:", (8, 18),
             (180,200,240), 0.45)

    # Width squares
    start_x = 12
    for i in range(full_w):
        cv2.rectangle(count_canvas,
                      (start_x + i*sq_viz_size, 28),
                      (start_x + (i+1)*sq_viz_size, 28+sq_viz_size),
                      (60,200,120), -1)
        cv2.rectangle(count_canvas,
                      (start_x + i*sq_viz_size, 28),
                      (start_x + (i+1)*sq_viz_size, 28+sq_viz_size),
                      (30,120,70), 1)
    if frac_w > 0.01:
        fx = start_x + full_w*sq_viz_size
        fw = int(frac_w*sq_viz_size)
        cv2.rectangle(count_canvas, (fx, 28), (fx+fw, 28+sq_viz_size),
                      (60,200,120), -1)
        cv2.rectangle(count_canvas, (fx, 28),
                      (fx+sq_viz_size, 28+sq_viz_size), (30,120,70), 1)

    put_text(count_canvas,
             f"Width: {full_w} full + {frac_w:.2f} = {obj_squares_w:.2f} squares × {grid_cm}cm = {real_w:.3f} cm",
             (12, 28+sq_viz_size+16), (100,240,160), 0.42)

    # Height squares
    start_y = 28+sq_viz_size+30
    for i in range(full_h):
        cv2.rectangle(count_canvas,
                      (12, start_y + i*sq_viz_size),
                      (12+sq_viz_size, start_y + (i+1)*sq_viz_size),
                      (80,140,220), -1)
        cv2.rectangle(count_canvas,
                      (12, start_y + i*sq_viz_size),
                      (12+sq_viz_size, start_y + (i+1)*sq_viz_size),
                      (40,80,160), 1)
    if frac_h > 0.01:
        fy = start_y + full_h*sq_viz_size
        fh = int(frac_h*sq_viz_size)
        cv2.rectangle(count_canvas, (12, fy), (12+sq_viz_size, fy+fh),
                      (80,140,220), -1)
        cv2.rectangle(count_canvas, (12, fy),
                      (12+sq_viz_size, start_y+(full_h+1)*sq_viz_size),
                      (40,80,160), 1)

    put_text(count_canvas,
             f"Height: {full_h} full + {frac_h:.2f} = {obj_squares_h:.2f} squares × {grid_cm}cm = {real_h:.3f} cm",
             (12+sq_viz_size+8, start_y+sq_viz_size//2+4), (80,160,240), 0.42)

    # ── combine main vis + count panel ────────────────────────────────────
    combined = np.vstack([vis, count_canvas])

    save_result(combined, "result")

    # ── metrics ───────────────────────────────────────────────────────────
    metrics = {
        "Grid square size (cm)":         f"{grid_cm} cm",
        "Grid square size (pixels)":      f"{grid_px} px",
        "Scale factor":                   f"{scale:.6f} cm/pixel",
        "─────────────────────────────": "─────────────────",
        "Object X (top-left px)":         f"{ox}",
        "Object Y (top-left px)":         f"{oy}",
        "Object width (pixels)":          f"{ow} px",
        "Object height (pixels)":         f"{oh} px",
        "─────────────────────────────ᵇ": "─────────────────",
        "Object width (squares)":         f"{obj_squares_w:.4f}",
        "Object height (squares)":        f"{obj_squares_h:.4f}",
        "─────────────────────────────ᶜ": "─────────────────",
        "REAL WIDTH (cm)":                f"► {real_w:.4f} cm",
        "REAL HEIGHT (cm)":               f"► {real_h:.4f} cm",
        "REAL AREA (cm²)":                f"► {real_area:.4f} cm²",
        "Diagonal (cm)":                  f"{diagonal_cm:.4f} cm",
        "Perimeter estimate (cm)":        f"{perimeter_cm:.4f} cm",
        "─────────────────────────────ᵈ": "─────────────────",
        "Formula (width)":                "squares_w × grid_cm",
        "Formula (area)":                 "squares_w × squares_h × grid_cm²",
    }
    show_metrics(metrics)
    return metrics, combined

# ── main entry point ──────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Method 8: Grid Reference Measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument("--image",
                    default="images/method8_grid/demo.jpg",
                    help="Path to input image")
    ap.add_argument("--grid_cm",
                    type=float, default=1.0,
                    help="Real size of one grid square in cm (default: 1.0)")
    ap.add_argument("--grid_px",
                    type=int, default=40,
                    help="Pixel size of one grid square (default: 40)")
    ap.add_argument("--obj_squares_w",
                    type=float, default=None,
                    help="Object width in grid squares (e.g. 3.5). Overrides pixel measurement.")
    ap.add_argument("--obj_squares_h",
                    type=float, default=None,
                    help="Object height in grid squares. Overrides pixel measurement.")
    ap.add_argument("--obj_x",
                    type=int, default=None,
                    help="Object bounding box top-left X (pixels)")
    ap.add_argument("--obj_y",
                    type=int, default=None,
                    help="Object bounding box top-left Y (pixels)")
    ap.add_argument("--obj_w",
                    type=int, default=None,
                    help="Object bounding box width (pixels)")
    ap.add_argument("--obj_h",
                    type=int, default=None,
                    help="Object bounding box height (pixels)")
    ap.add_argument("--interactive",
                    action="store_true",
                    help="Click object corners on image")
    ap.add_argument("--auto_detect",
                    action="store_true",
                    help="Auto-detect grid spacing from image")
    ap.add_argument("--generate_grid",
                    action="store_true",
                    help="Generate and save a printable A4 grid")
    ap.add_argument("--no_overlay",
                    action="store_true",
                    help="Disable grid overlay on output image")

    args = ap.parse_args()

    print("\n" + "═"*50)
    print("  METHOD 8 — Grid Reference Measurement")
    print("═"*50)

    # ── generate printable grid if requested ───────────────────────────────
    if args.generate_grid:
        generate_printable_grid(grid_cm=args.grid_cm)
        print("  Print print_this_grid.png at 100% scale on A4 paper.")
        print("  Place your object on it and photograph from above.")
        if not os.path.exists(args.image):
            return

    # ── create demo image if no image provided ─────────────────────────────
    auto_obj = None
    auto_sq_w = None
    auto_sq_h = None

    if not os.path.exists(args.image):
        print(f"\n  Image not found at '{args.image}'")
        print("  Generating synthetic demo image...\n")
        _, auto_obj, auto_sq_w, auto_sq_h = make_demo_image(
            grid_px=args.grid_px,
            grid_cm=args.grid_cm,
            out_path=args.image)

    img = load_image(args.image)
    h_img, w_img = img.shape[:2]
    print(f"\n  Image loaded: {args.image}  [{w_img}×{h_img} px]")

    # ── auto-detect grid spacing ───────────────────────────────────────────
    if args.auto_detect:
        detected_px, _ = detect_grid_spacing(img)
        if detected_px and detected_px > 3:
            args.grid_px = detected_px
            print(f"  Grid spacing updated to: {args.grid_px} px")
        else:
            print("  Auto-detect uncertain. Using --grid_px value.")

    # ── get object bounding box ────────────────────────────────────────────
    obj_x, obj_y, obj_w_px, obj_h_px = None, None, None, None

    if args.interactive:
        result = interactive_select(img)
        if result:
            obj_x, obj_y, obj_w_px, obj_h_px = result
            print(f"\n  Selected: x={obj_x} y={obj_y} w={obj_w_px} h={obj_h_px}")

    elif args.obj_x is not None:
        obj_x, obj_y = args.obj_x, args.obj_y
        obj_w_px = args.obj_w or int((args.obj_squares_w or 5) * args.grid_px)
        obj_h_px = args.obj_h or int((args.obj_squares_h or 3) * args.grid_px)

    elif auto_obj is not None:
        obj_x, obj_y, obj_w_px, obj_h_px = auto_obj

    else:
        # Default: centre region of image
        obj_w_px = int((args.obj_squares_w or 7) * args.grid_px)
        obj_h_px = int((args.obj_squares_h or 3) * args.grid_px)
        obj_x = (w_img - obj_w_px) // 2
        obj_y = (h_img - obj_h_px) // 2
        print(f"\n  No object coords given. Using centre: "
              f"x={obj_x} y={obj_y} w={obj_w_px} h={obj_h_px}")
        print("  Tip: use --interactive to click the object, or pass --obj_x --obj_y --obj_w --obj_h")

    # ── run measurement ────────────────────────────────────────────────────
    print(f"\n  Grid: {args.grid_cm} cm per square | {args.grid_px} px per square")
    print(f"  Object bounding box: x={obj_x} y={obj_y} w={obj_w_px} h={obj_h_px}")

    sq_w = args.obj_squares_w or auto_sq_w
    sq_h = args.obj_squares_h or auto_sq_h

    metrics, output = run(
        image=img,
        grid_cm=args.grid_cm,
        grid_px=args.grid_px,
        obj_x=obj_x,
        obj_y=obj_y,
        obj_w_px=obj_w_px,
        obj_h_px=obj_h_px,
        obj_squares_w=sq_w,
        obj_squares_h=sq_h,
        show_overlay=not args.no_overlay,
    )

    print("\n  Done. Check the results/ folder for output images.")

    # ── optional: display result ───────────────────────────────────────────
    try:
        small = output.copy()
        max_dim = 900
        sh, sw = small.shape[:2]
        if max(sh, sw) > max_dim:
            r = max_dim / max(sh, sw)
            small = cv2.resize(small, (int(sw*r), int(sh*r)))
        cv2.imshow("Method 8 — Grid Reference Result  (press Q to close)", small)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass  # headless environment — result is saved to disk


if __name__ == "__main__":
    main()