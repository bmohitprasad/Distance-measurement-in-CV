import cv2
import cv2.aruco as aruco
import numpy as np

def measure_scene(image_path, marker_size_real):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Marker (The "Ruler")
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    
    if ids is None:
        raise ValueError("Marker not found. I need the marker to set the scale!")

    # Calculate pixels-per-metric ratio from the marker
    c = corners[0][0]
    marker_width_px = np.linalg.norm(c[0] - c[1])
    pixels_per_metric = marker_width_px / marker_size_real
    print(f"Scale established: {pixels_per_metric:.2f} pixels per cm")

    # 2. Detect other objects (Contours)
    # Simple thresholding to find dark objects on light background
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = img.copy()
    
    for cnt in contours:
        # Ignore tiny noise
        if cv2.contourArea(cnt) < 500:
            continue
            
        # Get bounding box of the object
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Convert pixel size to real-world size using our marker ratio
        obj_width_cm = w / pixels_per_metric
        obj_height_cm = h / pixels_per_metric
        
        # Annotate
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(annotated, f"{obj_width_cm:.1f}x{obj_height_cm:.1f}cm", 
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite("a7_measured_scene.jpg", annotated)
    print("Measurement complete. Saved to a7_measured_scene.jpg")

if __name__ == "__main__":
    measure_scene("images/your_photo.jpg", 10.0) # 10.0 is marker size in cm