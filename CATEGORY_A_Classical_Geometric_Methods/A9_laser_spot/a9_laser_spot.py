import cv2
import argparse
import numpy as np
from utils.camera_utils import load_image, save_image, get_laser_mask

def detect_laser_dots(image_path, output_path=None):
    """
    Detects two laser dots and returns their (x, y) centers.
    """
    img = load_image(image_path)
    mask = get_laser_mask(img)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    dots = []
    annotated = img.copy()
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 10: continue
        
        # Get center of the dot
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < 2: continue
            
        dots.append((int(x), int(y)))
        
        # Annotate
        cv2.circle(annotated, (int(x), int(y)), int(radius)+5, (0, 255, 0), 2)
        cv2.putText(annotated, f"Dot {len(dots)}", (int(x)-10, int(y)-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if output_path:
        save_image(annotated, output_path)
        
    return dots

def distance_from_dual_laser(separation_px, baseline_real, focal_length_px):
    """
    Math: Distance = (Baseline * FocalLength) / Separation
    """
    if separation_px == 0: return float('inf')
    return (baseline_real * focal_length_px) / separation_px

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A9 - Laser Triangulation")
    parser.add_argument("--image", required=True)
    parser.add_argument("--baseline", type=float, required=True, help="Distance between lasers in cm")
    parser.add_argument("--focal", type=float, required=True, help="Camera focal length in pixels")
    parser.add_argument("--output", default="a9_result.jpg")
    
    args = parser.parse_args()
    
    dots = detect_laser_dots(args.image, args.output)
    
    if len(dots) == 2:
        sep = abs(dots[0][0] - dots[1][0])
        dist = distance_from_dual_laser(sep, args.baseline, args.focal)
        print(f"Detected 2 dots. Separation: {sep}px.")
        print(f"Calculated Distance: {dist:.2f} cm")
    else:
        print(f"Error: Found {len(dots)} dots. Need exactly 2 to calculate distance.")