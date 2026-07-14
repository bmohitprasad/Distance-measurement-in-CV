import cv2
import numpy as np
from utils.camera_utils import load_image

def show_contours(image_path):
    img = load_image(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, str(i), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
            
    cv2.imshow("Detected Contours", img)
    print("Press any key to close...")
    cv2.waitKey(0)

if __name__ == "__main__":
    show_contours("images/a8_object.jpg")