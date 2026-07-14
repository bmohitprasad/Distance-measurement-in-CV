"""
calibrate_webcam.py
-------------------
Use this script to find your camera's focal length and principal point.

1. Run the script to enter "Capture Mode": Press 's' to save 15-20 images 
   of your 10x7 checkerboard into a folder named 'calibration_images/'.
2. Run the script again to enter "Calibration Mode": It will process the
   saved images and calculate your camera's Intrinsic Matrix.
"""

import cv2
import numpy as np
import glob
import os

# Configuration for your 10x7 checkerboard
# 10x7 squares means 9 inner corners horizontally and 6 vertically
CHECKERBOARD_DIMS = (9, 6)

def capture_images():
    if not os.path.exists('calibration_images'):
        os.makedirs('calibration_images')
        
    cap = cv2.VideoCapture(0)
    count = 0
    print("Capture Mode: Press 's' to save, 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1)
        
        if key & 0xFF == ord('s'):
            cv2.imwrite(f'calibration_images/calib_{count}.jpg', frame)
            print(f"Saved calibration_images/calib_{count}.jpg")
            count += 1
        elif key & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

def calibrate():
    images = glob.glob('calibration_images/*.jpg')
    if not images:
        print("No images found in 'calibration_images/'. Run in Capture Mode first.")
        return

    # Prepare object points, e.g., (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((CHECKERBOARD_DIMS[0] * CHECKERBOARD_DIMS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_DIMS[0], 0:CHECKERBOARD_DIMS[1]].T.reshape(-1, 2)

    objpoints = []
    imgpoints = []
    img_shape = None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray.shape[::-1]
            
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_DIMS, None)
        
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"Detected corners in: {fname}")

    if not objpoints:
        print("Error: No checkerboard corners detected in any image.")
        return

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

    print("\n--- CAMERA CALIBRATION RESULTS ---")
    print(f"Intrinsic Matrix (K):\n{mtx}")
    print(f"\nFocal Length fx: {mtx[0,0]:.2f}")
    print(f"Focal Length fy: {mtx[1,1]:.2f}")
    print(f"Principal Point cx: {mtx[0,2]:.2f}")
    print(f"Principal Point cy: {mtx[1,2]:.2f}")

if __name__ == "__main__":
    if not os.path.exists('calibration_images'):
        capture_images()
    else:
        calibrate()