#!/usr/bin/env python3
import os
import sys
import subprocess

def check_dependencies():
    try:
        import ultralytics, cv2, pandas, pytesseract, PIL, flask
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def install_dependencies():
    print("Installing dependencies...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', 
                   'ultralytics', 'opencv-python', 'pandas', 
                   'pytesseract', 'pillow', 'flask'])

def main():
    print("=== Tunnel YOLO ===")
    print("1. Train model")
    print("2. Test model (analyze images)")
    print("3. Web GUI (single image analysis)")
    print("4. Generate HTML report")
    print("5. Install dependencies")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        if not check_dependencies():
            install_dependencies()
        
        if os.path.exists('runs/detect/train/weights/best.pt'):
            retrain = input("Model exists. Retrain? (y/n): ").lower() == 'y'
            if not retrain:
                print("Using existing model")
                return
        
        print("Starting training...")
        subprocess.run([sys.executable, 'train.py'])
        
    elif choice == "2":
        if not os.path.exists('runs/detect/train/weights/best.pt'):
            print("No trained model found. Run training first.")
            return
            
        if not check_dependencies():
            install_dependencies()
            
        print("Analyzing images...")
        subprocess.run([sys.executable, 'test.py'])
        print("Results saved to test-image-results.csv")
        
    elif choice == "3":
        if not os.path.exists('runs/detect/train/weights/best.pt'):
            print("No trained model found. Run training first.")
            return
            
        print("Starting web GUI...")
        print("Opening browser at http://localhost:8501")
        subprocess.run([sys.executable, 'flask_gui.py'])
        
    elif choice == "4":
        if not os.path.exists('runs/detect/train/weights/best.pt'):
            print("No trained model found. Run training first.")
            return
            
        if not check_dependencies():
            install_dependencies()
            
        print("Generating HTML report...")
        subprocess.run([sys.executable, 'report.py'])
        
    elif choice == "5":
        install_dependencies()
        print("Dependencies installed")
        
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()