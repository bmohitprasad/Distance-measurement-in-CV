"""
d1_numpy_regression.py
----------------------
Uses OpenCV (cv2) and NumPy to estimate distance via color thresholding.
No AI models used.
"""

import cv2
import numpy as np
import argparse

def estimate_distance(image_path, known_width_cm, focal_length_px):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Could not load image.")
        return

    # Convert to HSV (Hue, Saturation, Value) - Much better for color filtering than RGB
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define color range for the teal/blue car
    # You can tune these values!
    lower_blue = np.array([80, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No object detected. Try adjusting the color range in the script.")
        return

    # Find the largest contour (the car)
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Calculate Distance: D = (W_real * Focal) / W_pixel
    distance_cm = (known_width_cm * focal_length_px) / w
    distance_m = distance_cm / 100

    # Annotate image
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, f"Dist: {distance_m:.2f}m", (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite("d1_result.jpg", img)
    print(f"Detected Car. Width: {w}px, Distance: {distance_m:.2f} meters")
    cv2.imshow("Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--width", type=float, default=10.0, help="Car width in cm")
    parser.add_argument("--focal", type=float, default=1400.0, help="Calibration focal length")
    args = parser.parse_args()
    
    estimate_distance(args.image, args.width, args.focal)