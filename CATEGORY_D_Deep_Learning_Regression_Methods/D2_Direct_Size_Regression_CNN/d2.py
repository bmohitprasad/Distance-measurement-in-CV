"""
d2_size_regression.py
---------------------
Category D2: Direct Size Regression CNN
Uses a truncated ResNet-18 backbone with a custom linear regression head 
to predict real-world physical dimensions (cm) directly from image crops.
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ============================================================================
# SECTION 1 — SYNTHETIC DATASET FOR DEMO PURPOSES
# ============================================================================

class SyntheticSizeDataset(Dataset):
    """Generates synthetic rectangles of varying sizes to test regression convergence."""
    def __init__(self, num_samples=200, img_size=224):
        self.num_samples = num_samples
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a blank image background
        img_np = np.full((self.img_size, self.img_size, 3), 200, dtype=np.uint8)
        
        # Random physical width in cm (e.g., between 5.0cm and 25.0cm)
        true_width_cm = float(np.random.uniform(5.0, 25.0))
        
        # Pixel width is proportional to true width in this synthetic space
        box_size_px = int(true_width_cm * 6)
        
        x1 = (self.img_size - box_size_px) // 2
        y1 = (self.img_size - box_size_px) // 2
        x2 = x1 + box_size_px
        y2 = y1 + box_size_px

        # Draw object
        color = np.random.randint(50, 150, 3).tolist()
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, -1)
        
        # Add slight noise
        noise = np.random.randint(-15, 15, img_np.shape, dtype=np.int16)
        img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        img_tensor = self.transform(img_pil)
        label_tensor = torch.tensor(true_width_cm, dtype=torch.float32)

        return img_tensor, label_tensor


# ============================================================================
# SECTION 2 — REAL CUSTOM DATASET
# ============================================================================

class CustomObjectSizeDataset(Dataset):
    """Loads user-captured image crops and a CSV file mapping filenames to true widths in cm."""
    def __init__(self, csv_file, img_dir):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        
        image_tensor = self.transform(image)
        width_cm = torch.tensor(float(row['width_cm']), dtype=torch.float32)
        
        return image_tensor, width_cm


# ============================================================================
# SECTION 3 — SIZE REGRESSOR CNN ARCHITECTURE
# ============================================================================

class SizeRegressorCNN(nn.Module):
    """Truncated ResNet-18 with a custom single-neuron linear regression head."""
    def __init__(self):
        super(SizeRegressorCNN, self).__init__()
        # Load standard pre-trained ResNet-18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Strip off the final classification layer (fc), keep feature extractor
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # Replace with a regression head mapping 512 features to 1 continuous scalar
        self.regressor = nn.Linear(512, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        out = self.regressor(x)
        return out.squeeze(-1)


# ============================================================================
# SECTION 4 — TRAINING & INFERENCE ROUTINES
# ============================================================================

def train_model(dataset, epochs=10, batch_size=16, lr=1e-4, save_path="d2_weights.pth"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SizeRegressorCNN()
    criterion = nn.SmoothL1Loss()  # Robust regression loss
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\nTraining SizeRegressorCNN for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        total_loss = epoch_loss / len(dataset)
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1}/{epochs}] - Loss (Smooth L1): {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Training complete. Weights saved to -> {save_path}")
    return model


def predict_size(model_path, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SizeRegressorCNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Warning: Weights file '{model_path}' not found. Using uninitialized model.")
    
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predicted_width = model(input_tensor).item()

    print(f"\n--- D2 Size Regression Inference ---")
    print(f"Image File: {image_path}")
    print(f"Predicted Real-World Width: {predicted_width:.2f} cm")
    return predicted_width


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Category D2: Direct Size Regression CNN")
    parser.add_argument("--demo", action="store_true", help="Run a quick synthetic training & inference demo")
    parser.add_argument("--train", action="store_true", help="Train on custom image dataset")
    parser.add_argument("--data_dir", type=str, help="Directory containing image crops")
    parser.add_argument("--labels", type=str, help="CSV file mapping filename to width_cm")
    parser.add_argument("--image", type=str, help="Single image path for inference")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--load_weights", type=str, default="d2_weights.pth", help="Path to weights file")
    
    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    if args.demo:
        print("Running D2 Synthetic Training Demo...")
        synthetic_dataset = SyntheticSizeDataset(num_samples=150)
        model = train_model(synthetic_dataset, epochs=args.epochs, save_path=args.load_weights)
        
        # Generate a synthetic test sample for verification
        test_img, true_val = synthetic_dataset[0]
        model.eval()
        with torch.no_grad():
            pred_val = model(test_img.unsqueeze(0)).item()
        print(f"\n[Demo Results] Ground Truth Width: {true_val.item():.2f} cm | Predicted Width: {pred_val:.2f} cm")
        return

    if args.train:
        if not args.data_dir or not args.labels:
            print("Error: --train requires both --data_dir and --labels CSV specified.")
            return
        dataset = CustomObjectSizeDataset(csv_file=args.labels, img_dir=args.data_dir)
        train_model(dataset, epochs=args.epochs, save_path=args.load_weights)
        return

    if args.image:
        predict_size(args.load_weights, args.image)
        return

    parser.print_help()

if __name__ == "__main__":
    main()