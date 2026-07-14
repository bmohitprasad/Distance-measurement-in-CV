import cv2
import cv2.aruco as aruco
import numpy as np
import argparse

def get_scale_from_aruco(img, marker_size_real):
    """Detects ArUco marker and returns the pixels-per-cm ratio."""
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    
    if ids is None:
        return None, None, None
    
    # Use the first marker found to establish scale
    c = corners[0][0]
    marker_width_px = np.linalg.norm(c[0] - c[1])
    pixels_per_cm = marker_width_px / marker_size_real
    return pixels_per_cm, corners, ids

def process_scene(image_path, marker_size_real, output_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return

    # 1. Establish Scale via ArUco
    pixels_per_cm, corners, ids = get_scale_from_aruco(img, marker_size_real)
    if pixels_per_cm is None:
        print("Error: No ArUco marker detected. Cannot establish scale.")
        return

    print(f"Scale established: {pixels_per_cm:.2f} pixels per cm")

    # 2. Pre-processing for better edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Use Canny to detect edges
    edged = cv2.Canny(blurred, 30, 100)
    
    # Robust Morphological Closing: This "connects the dots" of fragmented edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = img.copy()
    if corners is not None:
        aruco.drawDetectedMarkers(annotated, corners, ids)
    
    print(f"\nScanning {len(contours)} contours for objects...")
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Calculate shape properties
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        
        # Diagnostic Printing
        print(f"Contour {i}: Area={area:.0f}, Corners={len(approx)}, Aspect={aspect_ratio:.2f}")

        # Check if this contour overlaps with our marker (ignore marker)
        if corners is not None:
            is_marker = False
            for corner in corners[0][0]:
                if x < corner[0] < x+w and y < corner[1] < y+h:
                    is_marker = True
            if is_marker: 
                print(" -> Skipped: It's the marker.")
                continue

        # Area filter: ignore very small noise (Adjust if book is missed)
        if area < 1000:
            print(" -> Skipped: Too small.")
            continue

        # Relaxed shape matching: Look for 4+ corners and a reasonable aspect ratio
        if 3 <= len(approx) <= 10 and 0.2 < aspect_ratio < 5.0:
            obj_width_cm = w / pixels_per_cm
            obj_height_cm = h / pixels_per_cm
            
            # Draw results
            cv2.drawContours(annotated, [approx], -1, (0, 255, 0), 3)
            cv2.putText(annotated, f"{obj_width_cm:.1f}x{obj_height_cm:.1f}cm", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f" -> MATCH! Object detected: {obj_width_cm:.1f}x{obj_height_cm:.1f}cm")
        else:
            print(f" -> Skipped: Failed shape/aspect filter (Area={area:.0f}, Aspect={aspect_ratio:.2f})")

    cv2.imwrite(output_path, annotated)
    print(f"\nMeasurement complete. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A7 Marker Main Script")
    parser.add_argument("--image", required=True)
    parser.add_argument("--marker-size", type=float, required=True)
    parser.add_argument("--output", default="a7_measured_scene.jpg")
    args = parser.parse_args()
    
    process_scene(args.image, args.marker_size, args.output)