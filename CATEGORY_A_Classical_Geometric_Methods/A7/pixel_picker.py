"""
pixel_picker.py
----------------
A simple tool to extract (x, y) coordinates.
1. Run this script.
2. Left-click 4 points on your image.
3. The coordinates will print to your terminal.
4. Press 'q' to close the window.
"""

import cv2

coords = []

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Captured: {x}, {y}")
        coords.append((x, y))
        # Draw a small circle to show you clicked it
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Pixel Picker", img)

# Load your image
image_path = "images/a7_marker.jpg" 
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not find {image_path}")
else:
    cv2.imshow("Pixel Picker", img)
    cv2.setMouseCallback("Pixel Picker", click_event)

    print("Click 4 points. Press 'q' when finished.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(coords) >= 4:
        print("\nCopy this for your A4 command:")
        # Format: "X1,Y1 X2,Y2 X3,Y3 X4,Y4"
        formatted = " ".join([f"{c[0]},{c[1]}" for c in coords])
        print(f'--img-pts "{formatted}"')