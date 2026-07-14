import os
import cv2

def load_image(image_path):
    """Loads an image from disk."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at '{image_path}'.")
    return img

def save_image(img, output_path):
    """Saves the image to the specified path."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path