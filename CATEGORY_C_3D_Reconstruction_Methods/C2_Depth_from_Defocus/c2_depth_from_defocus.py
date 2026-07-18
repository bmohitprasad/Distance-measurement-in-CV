"""
c3_depth_from_defocus.py
------------------------
Calculates a 'sharpness map' using the Variance of the Laplacian.
Higher values indicate areas in focus (likely the depth plane of the camera).
"""

import cv2
import numpy as np
import argparse

def compute_sharpness_map(image_path, window_size=30):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calculate Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # We use a sliding window to calculate variance locally
    # High variance = sharp, Low variance = blurry
    h, w = gray.shape
    sharpness_map = np.zeros((h, w), dtype=np.float32)
    
    for y in range(0, h, window_size):
        for x in range(0, w, window_size):
            roi = laplacian[y:y+window_size, x:x+window_size]
            variance = roi.var()
            sharpness_map[y:y+window_size, x:x+window_size] = variance
            
    return sharpness_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C3 Depth from Defocus")
    parser.add_argument("--image", required=True)
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--output", default="c3_sharpness_map.jpg")
    args = parser.parse_args()
    
    s_map = compute_sharpness_map(args.image, args.window)
    
    # Normalize for visualization
    s_map_norm = cv2.normalize(s_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap = cv2.applyColorMap(s_map_norm, cv2.COLORMAP_JET)
    
    cv2.imwrite(args.output, heatmap)
    print(f"Sharpness map saved to {args.output}. Warm colors = In Focus.")