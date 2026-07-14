import cv2
import numpy as np
import argparse
from utils.camera_utils import load_image, save_image

def measure_on_grid(image_path, cell_size_cm, output_path=None):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    tile_widths = []
    print(f"\nScanning {len(contours)} contours for grid tiles...")
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000: continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Grid tiles are squares (aspect ratio ~1.0)
        if 0.9 < aspect_ratio < 1.1:
            tile_widths.append(w)
            
    if not tile_widths:
        print("Error: Could not detect grid tiles. Try adjusting blur/canny settings.")
        return {"status": "error", "message": "No tiles detected"}
        
    avg_tile_px = np.mean(tile_widths)
    pixels_per_cm = avg_tile_px / cell_size_cm
    print(f"Calibration successful! Avg tile width: {avg_tile_px:.1f}px. Pixels-per-cm: {pixels_per_cm:.2f}")

    annotated = img.copy()
    processed_count = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1000: continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        
        # Filter out the grid tiles we used for calibration
        if 0.9 < aspect_ratio < 1.1:
            continue
            
        # If it's not a tile (i.e., not square), it's likely our object
        real_w = w / pixels_per_cm
        real_h = h / pixels_per_cm
        
        cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(annotated, f"{real_w:.1f}x{real_h:.1f}cm", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        processed_count += 1
        print(f"MATCH! Object detected: {real_w:.1f}x{real_h:.1f}cm (Aspect Ratio: {aspect_ratio:.2f})")
    
    if output_path:
        save_image(annotated, output_path)
        
    return {"status": "success", "processed_objects": processed_count}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A8 - Auto-Calibrating Grid Measurement")
    parser.add_argument("--image", required=True)
    parser.add_argument("--cell-size", type=float, default=30.0)
    parser.add_argument("--output", default="a8_result.jpg")
    args = parser.parse_args()
    
    print(measure_on_grid(args.image, args.cell_size, args.output))