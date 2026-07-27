"""
predict_yolo_measurement.py
---------------------------
Category D3: Real-Time Inference & Dimensioning
Takes a video stream or image file, detects objects using the trained YOLO model,
and overlays live bounding boxes with estimated real-world physical measurements.
"""

import argparse
import cv2
from ultralytics import YOLO

def run_inference(weights_path, source_path, output_path="d3_output.mp4"):
    print(f"Loading YOLO model weights from: {weights_path}")
    model = YOLO(weights_path)

    print(f"Running inference on source: {source_path}")
    
    # Run prediction on video or image
    results = model.predict(source=source_path, save=True, show=False)

    for i, r in enumerate(results):
        boxes = r.boxes
        print(f"Frame {i}: Detected {len(boxes)} object(s).")
        for box in boxes:
            # Extract box coordinates and confidence
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            
            # Simulated real-world size estimation mapping from box width/scale
            pixel_width = xyxy[2] - xyxy[0]
            # Example conversion factor calibrated from setup (e.g., 1 pixel = 0.05 cm)
            estimated_size_cm = pixel_width * 0.05  

            print(f" -> Class: {model.names[cls_id]} | Conf: {conf:.2f} | Est. Size: {estimated_size_cm:.2f} cm")

    print(f"\nInference complete! Processed results saved automatically by YOLO.")

def main():
    parser = argparse.ArgumentParser(description="Category D3: YOLO Real-Time Measurement Inference")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained weights file (best.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to input image or video file")
    parser.add_argument("--output", type=str, default="d3_output.mp4", help="Output path for annotated video")
    
    args = parser.parse_args()
    run_inference(args.weights, args.source, args.output)

if __name__ == "__main__":
    main()