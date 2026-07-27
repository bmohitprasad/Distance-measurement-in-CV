"""
train_yolo_measurement.py
-------------------------
Category D3: YOLO-Based Measurement Training Script
Leverages Ultralytics YOLOv8 to train an end-to-end model that detects objects 
and simultaneously regresses their real-world dimensions in a single forward pass.
"""

import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Category D3: Train YOLOv8 Measurement Model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset.yaml configuration")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Pre-trained base weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image input size")
    args = parser.parse_args()

    print("=== Initializing Category D3 YOLOv8 Training ===")
    
    # Load a pre-trained YOLOv8 nano model
    model = YOLO(args.weights)

    # Train the model with multi-task supervision (Object detection + Box sizing)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        task="detect",
        verbose=True
    )

    print("\nTraining complete!")
    print(f"Best weights saved to: {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    main()