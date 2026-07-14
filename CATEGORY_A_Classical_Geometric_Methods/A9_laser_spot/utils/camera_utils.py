import cv2
import numpy as np
import os

def load_image(image_path):
    """Loads an image for laser dot detection."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    return img

def save_image(img, output_path):
    """Saves the annotated image."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path

def get_laser_mask(img):
    """
    Isolates red laser dots using HSV color space.
    Laser light is extremely saturated and high value.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red range in HSV wraps around, so we create two masks
    mask1 = cv2.inRange(hsv, (0, 150, 200), (10, 255, 255))
    mask2 = cv2.inRange(hsv, (170, 150, 200), (180, 255, 255))
    
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Clean up noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask