"""
debug_a1.py
A simple script to visually debug what the Scale Reference method is seeing.
It draws bounding boxes around all detected contours and labels them with
their index (0 = leftmost, 1 = next, etc.) so you can see which object
the algorithm selected as the reference.
"""

import cv2
from utils.camera_utils import load_image, find_contours, min_area_rect_dimensions

def debug_contours(image_path):
    img = load_image(image_path)
    # Make a copy to draw on
    output_img = img.copy()
    
    contours = find_contours(img)
    print(f"Found {len(contours)} contours.")
    
    for i, c in enumerate(contours):
        box, width_px, height_px = min_area_rect_dimensions(c)
        
        # Draw the rotated bounding box in bright green (BGR: 0, 255, 0)
        cv2.drawContours(output_img, [box.astype("int")], -1, (0, 255, 0), 3)
        
        # Put the index number next to the box in red
        top_left = box[0]
        cv2.putText(output_img, f"Index {i}", (int(top_left[0]), int(top_left[1]) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        
        print(f"Index {i}: Width = {width_px:.1f} px, Height = {height_px:.1f} px")

    # Resize output for easy viewing on screen
    h, w = output_img.shape[:2]
    resized = cv2.resize(output_img, (w // 4, h // 4))
    
    cv2.imshow("Detected Contours", resized)
    print("Press any key on the image window to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Point this to your image
    debug_contours("images/a2_object.jpg")