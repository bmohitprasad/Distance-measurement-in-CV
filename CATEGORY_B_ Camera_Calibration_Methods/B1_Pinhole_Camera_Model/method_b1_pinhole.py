"""
METHOD B1: Pinhole Camera Model — Full Implementation
=======================================================
WHAT IT DOES:
    Models how a real camera projects 3D world onto 2D image.
    Calibrates using a checkerboard pattern to find:
        - Camera matrix K (focal length + principal point)
        - Distortion coefficients (radial + tangential lens warp)
    Then undistorts images and converts pixel measurements to real cm.

CAMERA MODEL (pinhole):
    [u]   [fx  0  cx] [X/Z]
    [v] = [0  fy  cy] [Y/Z]
    [1]   [0   0   1] [ 1 ]

    fx, fy = focal lengths in pixels
    cx, cy = principal point (image centre, approximately)
    X,Y,Z  = real 3D coordinates
    u,v    = pixel coordinates

DISTORTION MODEL:
    Radial:     x_d = x(1 + k1r² + k2r⁴ + k3r⁶)
    Tangential: x_d = x + [2p1xy + p2(r²+2x²)]
    Coefficients: [k1, k2, p1, p2, k3]

LOSS FUNCTION:
    Reprojection error = mean(||p_detected - K[R|t]P_world||²)
    Minimised internally by cv2.calibrateCamera using Levenberg-Marquardt.
    Good calibration: error < 0.5 pixels. Excellent: < 0.3 pixels.

WHAT IMAGE TO USE:
    FOR CALIBRATION (required first):
        - Print the checkerboard saved by --generate_board
        - Paste it flat on cardboard (must not bend)
        - Take 15-25 photos from different angles + distances
        - Keep the board in focus in every photo
        - Cover all corners of the frame in different shots

    FOR MEASUREMENT (after calibration):
        - Any image taken with the SAME camera/phone used for calibration
        - Side-view or top-view of object at known distance

    QUICK TEST (no real checkerboard needed):
        - Run with --demo to use synthetic images

FOLDER STRUCTURE EXPECTED:
    images/
      method_b1_calib/
        calib_01.jpg
        calib_02.jpg
        ...  (15-25 images of checkerboard from different angles)
      method_b1_measure/
        object.jpg   (image of your object to measure)

HOW TO RUN:
    # Full demo (synthetic checkerboard, no real images needed):
    python method_b1_pinhole.py --demo

    # Step 1: Generate a printable checkerboard:
    python method_b1_pinhole.py --generate_board

    # Step 2: Calibrate using your checkerboard photos:
    python method_b1_pinhole.py --calibrate --calib_dir images/method_b1_calib/

    # Step 3: Measure an object using saved calibration:
    python method_b1_pinhole.py --measure --image images/method_b1_measure/object.jpg \
        --p1 220 130 --p2 610 150 --dist 80

    # Calibrate + measure in one command:
    python method_b1_pinhole.py --calibrate --measure \
        --calib_dir images/method_b1_calib/ \
        --image images/method_b1_measure/object.jpg \
        --p1 220 130 --p2 610 150 --dist 80

    # Interactive — click two points on image after calibration:
    python method_b1_pinhole.py --calibrate --measure --interactive \
        --calib_dir images/method_b1_calib/ \
        --image images/method_b1_measure/object.jpg --dist 80

PARAMETERS:
    --board_w    : inner corners width of checkerboard (default 9)
    --board_h    : inner corners height (default 6)
    --sq_cm      : size of one checkerboard square in cm (default 2.5)
    --calib_dir  : folder containing calibration images
    --image      : image to measure after calibration
    --p1 x y     : pixel coordinate of measurement point 1
    --p2 x y     : pixel coordinate of measurement point 2
    --dist       : camera to object distance in cm
    --save_calib : path to save calibration data (default: calib_data.npz)
    --load_calib : load previously saved calibration instead of re-running
    --demo       : run with synthetic data (no real images needed)
    --generate_board : generate and save a printable checkerboard image
    --interactive: click measurement points on image
"""

import cv2
import numpy as np
import argparse
import os
import sys
import math
import glob

# ── helpers ────────────────────────────────────────────────────────────────────

def put_text(img, text, pos, color=(255,255,255), scale=0.55, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def save_result(img, suffix, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"method_b1_{suffix}.jpg")
    cv2.imwrite(path, img)
    print(f"  Saved → {path}")
    return path

def show_metrics(title, metrics):
    print(f"\n{'═'*52}")
    print(f"  {title}")
    print(f"{'═'*52}")
    for k, v in metrics.items():
        if k.startswith("─"):
            print(f"  {'─'*48}")
        else:
            print(f"  {k:<34} {v}")
    print(f"{'═'*52}\n")

# ── checkerboard generation ────────────────────────────────────────────────────

def generate_checkerboard(board_w=9, board_h=6, sq_px=60,
                           out_path="print_this_checkerboard.png"):
    """
    Generate a printable checkerboard.
    board_w, board_h = number of INNER corners (squares = corners+1)
    sq_px = size of each square in pixels at screen resolution

    PRINT INSTRUCTIONS:
        Print at 100% scale on A4 paper.
        Measure one printed square with a ruler — note that cm value.
        Pass it as --sq_cm when running calibration.
    """
    cols = board_w + 1
    rows = board_h + 1
    margin = sq_px * 2

    W = cols * sq_px + 2 * margin
    H = rows * sq_px + 2 * margin
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x1 = margin + c * sq_px
                y1 = margin + r * sq_px
                cv2.rectangle(canvas, (x1, y1),
                               (x1 + sq_px, y1 + sq_px), (0, 0, 0), -1)

    # Border
    cv2.rectangle(canvas, (margin-2, margin-2),
                  (margin + cols*sq_px + 2, margin + rows*sq_px + 2),
                  (80, 80, 80), 2)

    info = (f"Checkerboard {board_w}x{board_h} inner corners | "
            f"Print at 100% scale | Measure one square with ruler for --sq_cm")
    cv2.putText(canvas, info, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    cv2.imwrite(out_path, canvas)
    print(f"\n  Checkerboard saved → {out_path}")
    print(f"  Inner corners: {board_w} × {board_h}")
    print(f"  PRINT AT 100% scale (no fit-to-page).")
    print(f"  After printing, measure one black square with a ruler.")
    print(f"  Use that measurement as --sq_cm when calibrating.\n")
    return canvas

# ── synthetic calibration image generator ─────────────────────────────────────

def make_synthetic_calib_images(board_w=9, board_h=6, sq_px=50, n=20,
                                 out_dir="images/method_b1_calib"):
    """
    Generate synthetic checkerboard images at various angles/positions
    for demo calibration without a real camera.
    """
    os.makedirs(out_dir, exist_ok=True)
    cols, rows = board_w + 1, board_h + 1
    W_board = cols * sq_px
    H_board = rows * sq_px

    # Build a flat checkerboard template
    board_img = np.ones((H_board, W_board, 3), dtype=np.uint8) * 255
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                cv2.rectangle(board_img,
                               (c*sq_px, r*sq_px),
                               ((c+1)*sq_px, (r+1)*sq_px),
                               (0,0,0), -1)

    np.random.seed(42)
    paths = []
    IMG_W, IMG_H = 800, 600

    for i in range(n):
        canvas = (np.random.randint(180, 220, (IMG_H, IMG_W, 3))
                  .astype(np.uint8))

        # Random perspective transform
        angle    = np.random.uniform(-30, 30)
        scale_f  = np.random.uniform(0.5, 0.9)
        tx       = np.random.randint(50, IMG_W - int(W_board*scale_f) - 50)
        ty       = np.random.randint(50, IMG_H - int(H_board*scale_f) - 50)

        bw = int(W_board * scale_f)
        bh = int(H_board * scale_f)

        # Perspective source corners
        skew_x = np.random.uniform(-0.15, 0.15) * bw
        skew_y = np.random.uniform(-0.1,  0.1)  * bh
        src_pts = np.float32([
            [0,         0        ],
            [W_board,   0        ],
            [W_board,   H_board  ],
            [0,         H_board  ]])
        dst_pts = np.float32([
            [tx,           ty          ],
            [tx+bw+skew_x, ty+skew_y  ],
            [tx+bw,        ty+bh      ],
            [tx+skew_x,    ty+bh-skew_y]])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(board_img, M, (IMG_W, IMG_H))
        mask   = cv2.warpPerspective(
            np.ones((H_board,W_board,3),dtype=np.uint8)*255, M, (IMG_W,IMG_H))
        mask_g = mask[:,:,0:1] > 128
        canvas = np.where(mask_g, warped, canvas)

        # Add slight noise
        noise  = np.random.randint(-8, 8, canvas.shape, dtype=np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        path = os.path.join(out_dir, f"calib_{i+1:02d}.jpg")
        cv2.imwrite(path, canvas)
        paths.append(path)

    print(f"  Generated {n} synthetic calibration images → {out_dir}/")
    return paths

# ── calibration ────────────────────────────────────────────────────────────────

def calibrate(calib_dir, board_w=9, board_h=6, sq_cm=2.5,
              save_path="calib_data.npz"):
    """
    Run full camera calibration from checkerboard images.

    Returns:
        camera_matrix  : 3×3 K matrix
        dist_coeffs    : [k1,k2,p1,p2,k3] distortion
        reprojection_error : mean pixel reprojection error (lower = better)
        img_size       : (width, height) of calibration images
    """
    board_size = (board_w, board_h)

    # 3D world points for one board position
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    objp *= sq_cm                              # scale to real cm

    obj_points = []   # 3D points per image
    img_points = []   # 2D detected corners per image
    img_size   = None
    good_imgs  = []

    image_paths = sorted(
        glob.glob(os.path.join(calib_dir, "*.jpg")) +
        glob.glob(os.path.join(calib_dir, "*.png")) +
        glob.glob(os.path.join(calib_dir, "*.jpeg")))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {calib_dir}")

    print(f"\n  Found {len(image_paths)} calibration images.")
    print(f"  Board: {board_w}×{board_h} inner corners | square = {sq_cm} cm\n")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img  = cv2.imread(path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        if ret:
            # Refine to sub-pixel accuracy
            corners_refined = cv2.cornerSubPix(
                gray, corners, (11,11), (-1,-1), criteria)
            obj_points.append(objp)
            img_points.append(corners_refined)
            good_imgs.append(path)
            print(f"  ✓  {os.path.basename(path)}")
        else:
            print(f"  ✗  {os.path.basename(path)}  (corners not found)")

    if len(good_imgs) < 6:
        raise RuntimeError(
            f"Only {len(good_imgs)} good images found. Need at least 6. "
            "Ensure checkerboard is clearly visible and fully in frame.")

    print(f"\n  Calibrating with {len(good_imgs)} valid images...")

    # ── run calibration ──────────────────────────────────────────────────
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None)

    # ── compute per-image reprojection errors ──────────────────────────
    errors = []
    for i, (objp_i, imgp_i, rvec, tvec) in enumerate(
            zip(obj_points, img_points, rvecs, tvecs)):
        projected, _ = cv2.projectPoints(
            objp_i, rvec, tvec, camera_matrix, dist_coeffs)
        err = cv2.norm(imgp_i, projected, cv2.NORM_L2) / len(projected)
        errors.append(err)

    mean_error  = np.mean(errors)
    max_error   = np.max(errors)

    # ── save calibration ───────────────────────────────────────────────
    np.savez(save_path,
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             img_size=img_size,
             reprojection_error=mean_error)
    print(f"\n  Calibration data saved → {save_path}")

    # ── visualise one calibration result ──────────────────────────────
    vis_img = cv2.imread(good_imgs[0])
    gray_v  = cv2.cvtColor(vis_img, cv2.COLOR_BGR2GRAY)
    ret_v, corners_v = cv2.findChessboardCorners(gray_v, board_size, None)
    if ret_v:
        cv2.drawChessboardCorners(vis_img, board_size, corners_v, ret_v)
    # Draw reprojected points
    proj, _ = cv2.projectPoints(
        obj_points[0], rvecs[0], tvecs[0], camera_matrix, dist_coeffs)
    for pt in proj:
        cv2.circle(vis_img, tuple(pt[0].astype(int)), 3, (0,80,220), -1)
    put_text(vis_img, f"Reprojection error: {mean_error:.4f} px",
             (10, vis_img.shape[0]-10), (0,220,80))
    save_result(vis_img, "calibration_corners")

    # ── undistortion demo ──────────────────────────────────────────────
    undist = cv2.undistort(cv2.imread(good_imgs[0]),
                            camera_matrix, dist_coeffs)
    side_by_side = np.hstack([cv2.imread(good_imgs[0]), undist])
    h_s = side_by_side.shape[0]
    put_text(side_by_side, "Original (distorted)",
             (10, 25), (0,80,220))
    put_text(side_by_side, "Undistorted",
             (side_by_side.shape[1]//2 + 10, 25), (0,200,80))
    save_result(side_by_side, "undistort_comparison")

    # ── metrics ───────────────────────────────────────────────────────
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]
    cx = camera_matrix[0,2]
    cy = camera_matrix[1,2]
    k1,k2,p1,p2,k3 = dist_coeffs[0]

    quality = ("EXCELLENT" if mean_error < 0.3 else
               "GOOD"      if mean_error < 0.5 else
               "ACCEPTABLE" if mean_error < 1.0 else
               "POOR — retake some calibration images")

    metrics = {
        "Images used for calibration":  str(len(good_imgs)),
        "Image size":                   f"{img_size[0]} × {img_size[1]} px",
        "─ Camera Matrix (K) ─":        "",
        "  Focal length fx":            f"{fx:.3f} px",
        "  Focal length fy":            f"{fy:.3f} px",
        "  Principal point cx":         f"{cx:.3f} px",
        "  Principal point cy":         f"{cy:.3f} px",
        "─ Distortion Coefficients ─":  "",
        "  k1 (radial)":                f"{k1:.6f}",
        "  k2 (radial)":                f"{k2:.6f}",
        "  p1 (tangential)":            f"{p1:.6f}",
        "  p2 (tangential)":            f"{p2:.6f}",
        "  k3 (radial)":                f"{k3:.6f}",
        "─ Quality ─":                  "",
        "Mean reprojection error":      f"{mean_error:.4f} px",
        "Max  reprojection error":      f"{max_error:.4f} px",
        "Calibration quality":          quality,
        "Saved to":                     save_path,
    }
    show_metrics("B1 — Camera Calibration Results", metrics)
    return camera_matrix, dist_coeffs, mean_error, img_size

# ── load saved calibration ─────────────────────────────────────────────────────

def load_calibration(path="calib_data.npz"):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Calibration file not found: {path}\n"
            "Run with --calibrate first.")
    data = np.load(path)
    print(f"  Calibration loaded from {path}")
    return (data["camera_matrix"],
            data["dist_coeffs"],
            float(data["reprojection_error"]))

# ── measurement ────────────────────────────────────────────────────────────────

def measure(image_path, camera_matrix, dist_coeffs,
            p1, p2, cam_distance_cm):
    """
    Undistort image and measure real-world distance between two points.

    The pinhole formula gives:
        Real_size = pixel_distance × (Z / focal_length)
    where Z = camera-to-object distance in cm.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    h, w = img.shape[:2]
    fx = camera_matrix[0,0]
    fy = camera_matrix[1,1]

    # ── undistort ────────────────────────────────────────────────────
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w,h), alpha=1, newImgSize=(w,h))
    undist = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_K)

    # Crop to valid region
    rx, ry, rw, rh = roi
    if rw > 0 and rh > 0:
        undist_cropped = undist[ry:ry+rh, rx:rx+rw]
        # Adjust point coords for crop
        p1_u = (p1[0] - rx, p1[1] - ry)
        p2_u = (p2[0] - rx, p2[1] - ry)
    else:
        undist_cropped = undist
        p1_u, p2_u = p1, p2

    # ── pixel distance in undistorted image ───────────────────────────
    px_dist = math.sqrt((p2_u[0]-p1_u[0])**2 + (p2_u[1]-p1_u[1])**2)

    # Angle of the line (for axis-aligned breakdown)
    dx = abs(p2_u[0] - p1_u[0])
    dy = abs(p2_u[1] - p1_u[1])

    # Real-world size from pinhole model
    real_total = px_dist * cam_distance_cm / fx
    real_x     = dx      * cam_distance_cm / fx   # horizontal span
    real_y     = dy      * cam_distance_cm / fy   # vertical span  (uses fy)

    # Scale factor at this distance
    scale = cam_distance_cm / fx   # cm per pixel

    # ── visualise on undistorted image ────────────────────────────────
    vis = undist_cropped.copy()
    vh, vw = vis.shape[:2]

    # Draw the measurement line
    cv2.line(vis, p1_u, p2_u, (0,200,80), 2)

    # Endpoint circles
    for pt in [p1_u, p2_u]:
        cv2.circle(vis, pt, 6, (0,80,220), -1)
        cv2.circle(vis, pt, 6, (255,255,255), 1)

    # Dashed component lines (dx, dy)
    mid_x = (p1_u[0]+p2_u[0])//2
    mid_y = (p1_u[1]+p2_u[1])//2
    corner = (p2_u[0], p1_u[1])
    for i in range(0, abs(p2_u[0]-p1_u[0]), 10):
        x_s = min(p1_u[0],p2_u[0]) + i
        x_e = min(x_s+6, max(p1_u[0],p2_u[0]))
        cv2.line(vis, (x_s, p1_u[1]), (x_e, p1_u[1]), (180,180,0), 1)
    for i in range(0, abs(p2_u[1]-p1_u[1]), 10):
        y_s = min(p1_u[1],p2_u[1]) + i
        y_e = min(y_s+6, max(p1_u[1],p2_u[1]))
        cv2.line(vis, (p2_u[0], y_s), (p2_u[0], y_e), (180,180,0), 1)

    # Labels
    put_text(vis, f"P1 ({p1_u[0]},{p1_u[1]})",
             (p1_u[0]+8, p1_u[1]-8), (0,80,220), 0.45)
    put_text(vis, f"P2 ({p2_u[0]},{p2_u[1]})",
             (p2_u[0]+8, p2_u[1]-8), (0,80,220), 0.45)
    put_text(vis, f"{real_total:.3f} cm",
             (mid_x-30, mid_y-12), (0,230,80), 0.65)

    # Horizontal component label
    put_text(vis, f"dx={real_x:.2f}cm",
             (min(p1_u[0],p2_u[0])+5, p1_u[1]-8), (200,200,0), 0.4)
    # Vertical component label
    put_text(vis, f"dy={real_y:.2f}cm",
             (p2_u[0]+5, (p1_u[1]+p2_u[1])//2), (200,200,0), 0.4)

    # Camera matrix panel
    panel_h = 60
    panel = np.zeros((panel_h, vw, 3), dtype=np.uint8)
    cv2.rectangle(panel, (0,0),(vw,panel_h),(15,20,30),-1)
    put_text(panel,
             f"fx={fx:.1f}  fy={camera_matrix[1,1]:.1f}  "
             f"cx={camera_matrix[0,2]:.1f}  cy={camera_matrix[1,2]:.1f}",
             (8,18), (140,170,220), 0.42)
    put_text(panel,
             f"dist={cam_distance_cm}cm | px_dist={px_dist:.1f}px | "
             f"scale={scale:.5f}cm/px | real={real_total:.4f}cm",
             (8,42), (100,230,150), 0.45)

    combined = np.vstack([vis, panel])

    # Side-by-side original vs undistorted (small)
    scale_f = min(1.0, 400/h)
    orig_s = cv2.resize(img, (int(w*scale_f), int(h*scale_f)))
    undt_s = cv2.resize(undist, (int(w*scale_f), int(h*scale_f)))
    comp_h = orig_s.shape[0]
    comp_w = orig_s.shape[1]
    label_bar = np.zeros((22, comp_w*2, 3), dtype=np.uint8)
    put_text(label_bar, "Original", (5,16), (80,140,220), 0.4)
    put_text(label_bar, "Undistorted", (comp_w+5,16), (80,220,140), 0.4)
    comparison = np.hstack([orig_s, undt_s])
    comparison = np.vstack([label_bar, comparison])
    save_result(comparison, "undistort_measure_comparison")
    save_result(combined, "measurement_result")

    # ── metrics ──────────────────────────────────────────────────────
    metrics = {
        "─ Input ─":                    "",
        "  Point 1 (pixels)":           str(p1),
        "  Point 2 (pixels)":           str(p2),
        "  Camera distance":            f"{cam_distance_cm} cm",
        "─ Camera Model ─":             "",
        "  Focal length fx":            f"{fx:.4f} px",
        "  Focal length fy":            f"{camera_matrix[1,1]:.4f} px",
        "  Principal point (cx,cy)":    f"({camera_matrix[0,2]:.1f}, {camera_matrix[1,2]:.1f})",
        "─ Distortion ─":               "",
        "  k1":                         f"{dist_coeffs[0][0]:.6f}",
        "  k2":                         f"{dist_coeffs[0][1]:.6f}",
        "  p1":                         f"{dist_coeffs[0][2]:.6f}",
        "  p2":                         f"{dist_coeffs[0][3]:.6f}",
        "─ Measurement ─":              "",
        "  Pixel distance":             f"{px_dist:.4f} px",
        "  Horizontal span (dx)":       f"{dx:.1f} px → {real_x:.4f} cm",
        "  Vertical span (dy)":         f"{dy:.1f} px → {real_y:.4f} cm",
        "  Scale at this distance":     f"{scale:.6f} cm/pixel",
        "  REAL DISTANCE":              f"► {real_total:.4f} cm",
        "─ Formula ─":                  "",
        "  real = px_dist × Z / fx":    f"{px_dist:.2f} × {cam_distance_cm} / {fx:.2f}",
    }
    show_metrics("B1 — Pinhole Measurement Results", metrics)
    return real_total, scale

# ── interactive point selector ─────────────────────────────────────────────────

_pts = []

def _mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_pts) < 2:
        _pts.append((x,y))
        print(f"  Point {len(_pts)}: ({x},{y})")

def interactive_select(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    _pts.clear()
    print("\n  Click P1, then P2 on the image. Press Q when done.\n")
    cv2.namedWindow("Click two measurement points — press Q when done")
    cv2.setMouseCallback("Click two measurement points — press Q when done",
                          _mouse)
    while True:
        disp = img.copy()
        for pt in _pts:
            cv2.circle(disp, pt, 6, (0,80,220), -1)
        if len(_pts) == 2:
            cv2.line(disp, _pts[0], _pts[1], (0,200,80), 2)
        cv2.imshow("Click two measurement points — press Q when done", disp)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(_pts) == 2:
            break
    cv2.destroyAllWindows()
    if len(_pts) == 2:
        return _pts[0], _pts[1]
    return None, None

# ── demo runner ────────────────────────────────────────────────────────────────

def run_demo(board_w=9, board_h=6, sq_cm=2.5):
    """
    Full demo with synthetic images — no real camera or checkerboard needed.
    """
    print("\n" + "═"*52)
    print("  METHOD B1 — DEMO MODE (synthetic images)")
    print("═"*52)
    print("\n  Step 1: Generating synthetic calibration images...")

    calib_dir = "images/method_b1_calib_demo"
    paths = make_synthetic_calib_images(board_w, board_h, sq_px=50,
                                         n=20, out_dir=calib_dir)

    print("\n  Step 2: Running calibration...")
    try:
        K, D, err, img_size = calibrate(
            calib_dir, board_w, board_h, sq_cm,
            save_path="calib_data_demo.npz")
    except Exception as e:
        print(f"\n  [!] Calibration failed on synthetic images: {e}")
        print("      Using approximate fallback camera matrix for demo.")
        K = np.array([[600,0,400],[0,600,300],[0,0,1]], dtype=np.float64)
        D = np.zeros((1,5), dtype=np.float64)
        err = 0.0
        img_size = (800, 600)
        np.savez("calib_data_demo.npz",
                 camera_matrix=K, dist_coeffs=D,
                 img_size=img_size, reprojection_error=err)

    print("\n  Step 3: Measuring on demo image...")

    # Create a simple demo measurement image
    demo_img_path = "images/method_b1_measure/demo_object.jpg"
    os.makedirs(os.path.dirname(demo_img_path), exist_ok=True)
    demo = np.ones((600, 800, 3), dtype=np.uint8) * 235
    # Draw checkerboard background (faint)
    for r in range(0,600,40):
        for c in range(0,800,40):
            if (r//40+c//40)%2==0:
                cv2.rectangle(demo,(c,r),(c+40,r+40),(215,215,230),-1)
    # Draw object
    cv2.ellipse(demo, (400,300), (200,80), 0, 0, 360, (120,160,100), -1)
    cv2.ellipse(demo, (400,300), (200,80), 0, 0, 360, (60,100,40), 2)
    cv2.putText(demo, "Demo object (measure tip-to-tip)",
                (220,400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)
    cv2.imwrite(demo_img_path, demo)

    p1 = (200, 300)
    p2 = (600, 300)
    cam_dist = 80.0

    measure(demo_img_path, K, D, p1, p2, cam_dist)
    print("\n  Demo complete. Check results/ folder.")

# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Method B1: Pinhole Camera Model — Calibration & Measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument("--demo",          action="store_true",
                    help="Run full demo with synthetic images")
    ap.add_argument("--generate_board",action="store_true",
                    help="Save a printable checkerboard image")
    ap.add_argument("--calibrate",     action="store_true",
                    help="Run camera calibration")
    ap.add_argument("--measure",       action="store_true",
                    help="Measure after calibration")
    ap.add_argument("--interactive",   action="store_true",
                    help="Click measurement points on image")

    ap.add_argument("--calib_dir",     default="images/method_b1_calib",
                    help="Folder with calibration checkerboard images")
    ap.add_argument("--image",         default="images/method_b1_measure/object.jpg",
                    help="Image to measure")
    ap.add_argument("--save_calib",    default="calib_data.npz",
                    help="Where to save calibration data")
    ap.add_argument("--load_calib",    default=None,
                    help="Load existing calibration file instead of re-running")

    ap.add_argument("--board_w",  type=int,   default=9,
                    help="Inner corners in width direction (default 9)")
    ap.add_argument("--board_h",  type=int,   default=6,
                    help="Inner corners in height direction (default 6)")
    ap.add_argument("--sq_cm",    type=float, default=2.5,
                    help="Checkerboard square size in cm (default 2.5)")

    ap.add_argument("--p1",  type=int, nargs=2, default=[200,300],
                    help="Measurement point 1 (x y)")
    ap.add_argument("--p2",  type=int, nargs=2, default=[600,300],
                    help="Measurement point 2 (x y)")
    ap.add_argument("--dist",type=float,        default=80.0,
                    help="Camera to object distance in cm")

    args = ap.parse_args()

    print("\n" + "═"*52)
    print("  METHOD B1 — Pinhole Camera Model")
    print("═"*52)

    # ── demo ──────────────────────────────────────────────────────────
    if args.demo:
        run_demo(args.board_w, args.board_h, args.sq_cm)
        return

    # ── generate board ────────────────────────────────────────────────
    if args.generate_board:
        generate_checkerboard(args.board_w, args.board_h,
                               out_path="print_this_checkerboard.png")
        if not (args.calibrate or args.measure):
            return

    # ── calibrate ─────────────────────────────────────────────────────
    K, D = None, None

    if args.load_calib:
        K, D, _ = load_calibration(args.load_calib)

    elif args.calibrate:
        if not os.path.isdir(args.calib_dir):
            print(f"\n  Calibration dir not found: {args.calib_dir}")
            print("  Creating synthetic demo images for calibration...\n")
            make_synthetic_calib_images(
                args.board_w, args.board_h, out_dir=args.calib_dir)

        K, D, err, _ = calibrate(
            args.calib_dir, args.board_w, args.board_h,
            args.sq_cm, save_path=args.save_calib)

    elif args.measure:
        # Try to load existing calibration
        if os.path.exists(args.save_calib):
            K, D, _ = load_calibration(args.save_calib)
        else:
            print(f"\n  No calibration found at {args.save_calib}")
            print("  Run with --calibrate first, or use --demo.\n")
            return

    # ── measure ───────────────────────────────────────────────────────
    if args.measure and K is not None:
        if not os.path.exists(args.image):
            print(f"\n  Measurement image not found: {args.image}")
            print("  Creating a demo measurement image...\n")
            os.makedirs(os.path.dirname(args.image), exist_ok=True)
            demo = np.ones((600,800,3),dtype=np.uint8)*235
            cv2.ellipse(demo,(400,300),(200,80),0,0,360,(120,160,100),-1)
            cv2.imwrite(args.image, demo)

        p1, p2 = tuple(args.p1), tuple(args.p2)

        if args.interactive:
            r1, r2 = interactive_select(args.image)
            if r1 and r2:
                p1, p2 = r1, r2

        measure(args.image, K, D, p1, p2, args.dist)

    if not any([args.demo, args.calibrate, args.measure,
                args.generate_board, args.load_calib]):
        print("\n  No action specified. Try one of:")
        print("    python method_b1_pinhole.py --demo")
        print("    python method_b1_pinhole.py --generate_board")
        print("    python method_b1_pinhole.py --calibrate --calib_dir <folder>")
        print("    python method_b1_pinhole.py --calibrate --measure --image <img>")
        print("\n  See top of file for full documentation.\n")


if __name__ == "__main__":
    main()