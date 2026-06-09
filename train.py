from ultralytics import YOLO
import os

# 1. Load the lightweight Nano model
model = YOLO('yolov8n.pt') 

# 2. Train FAST
# epochs=10: much quicker than 50
# imgsz=416: faster processing than 640
# workers=4: uses more CPU cores
print("🚀 Starting Fast Training (10 Epochs)...")
model.train(
    data='data.yaml', 
    epochs=10, 
    imgsz=416, 
    batch=16, 
    device='cpu', # Change to 0 if you have an NVIDIA GPU
    workers=4
)

print("✅ Training Complete!")
print("Your model is saved at: runs/detect/train/weights/best.pt")