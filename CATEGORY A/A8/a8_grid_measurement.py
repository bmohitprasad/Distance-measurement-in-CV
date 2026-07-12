import cv2
import numpy as np
import argparse
import json
from utils.camera_utils import load_image, save_image

def measure_on_grid(image_path, cell_size_cm, output_path=None):
    """
    Measures objects on a grid by calculating the pixels-per-cell ratio.
    """
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple edge detection to find the grid
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    
    # For a simple grid implementation, we approximate the pixels-per-cell 
    # based on the average distance between detected dominant horizontal/vertical lines.
    # In a production grid tool, we would cluster these lines.
    # For now, we assume a standard grid detection or manual override.
    
    # Find objects (blobs) on the grid
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated = img.copy()
    results = []
    
    # Simplified logic: 
    # Real-world ratio = pixels_per_cell / cell_size_cm
    # (Assuming we have calculated pixels_per_cell from line spacing)
    # Here we use a placeholder factor for demonstration - 
    # in usage, provide a clear grid with identifiable spacing.
    pixels_per_cell = 50.0  # Placeholder: Replace with actual detected grid spacing
    ratio = pixels_per_cell / cell_size_cm
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 100: continue
        x, y, w, h = cv2.boundingRect(cnt)
        
        real_w = w / ratio
        real_h = h / ratio
        
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(annotated, f"{real_w:.1f}x{real_h:.1f}cm", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    if output_path:
        save_image(annotated, output_path)
        
    return {"status": "success", "processed_objects": len(results)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A8 - Grid Measurement")
    parser.add_argument("--image", required=True)
    parser.add_argument("--cell-size", type=float, required=True, help="Size of one grid square in cm")
    parser.add_argument("--output", default="a8_result.jpg")
    args = parser.parse_args()
    
    print(measure_on_grid(args.image, args.cell_size, args.output))