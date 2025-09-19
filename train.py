#!/usr/bin/env python3
import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

def main():
    if os.path.exists('runs'):
        shutil.rmtree('runs')
    
    with open('train-images/classes.txt', 'r') as f:
        class_names = [line.strip() for line in f if line.strip()]
    
    yaml_data = {
        'path': str(Path('train-images').absolute()),
        'train': 'train/images',
        'val': 'validation/images',
        'nc': len(class_names),
        'names': class_names
    }
    
    with open('data.yaml', 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    
    print("Training YOLO model...")
    model = YOLO('yolo11s.pt')
    model.train(data='data.yaml', epochs=60, imgsz=640, batch=8, patience=10)
    
    print("Training completed!")
    print("Model saved to: runs/detect/train/weights/best.pt")

if __name__ == "__main__":
    main()