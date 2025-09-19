# Tunnel YOLO - Local Analysis System

A local Python application for automated analysis of RS2 tunnel model images using YOLOv11 object detection. This system detects tunnel features (crowns, labels, openings) and extracts displacement data through intelligent color matching.

## Features

- **Local execution** - No cloud dependencies, runs entirely on your machine
- **Modern web interface** - Dark theme Flask GUI for single image analysis
- **Batch processing** - Analyze multiple images via command line
- **HTML reports** - Professional reports with embedded images and displacement data
- **Clean architecture** - Minimal, organized Python scripts
- **Pre-trained model** - Ready-to-use YOLOv11 model included

## Quick Start

```bash
python run.py
```

Choose from 5 options:
1. **Train model** - Train on your own data
2. **Test model** - Batch analyze all test images
3. **Web GUI** - Interactive single image analysis
4. **Generate report** - Create HTML summary report
5. **Install dependencies** - Setup required packages

## File Structure

```
├── run.py              # Main interface
├── train.py            # Model training
├── test.py             # Batch image analysis
├── flask_gui.py        # Web interface
├── report.py           # HTML report generation
├── train-images/       # Training dataset
├── test-images/        # Test images (test-001.png, etc.)
└── runs/detect/train/  # Trained model and results
```

## Web Interface

Launch the modern web GUI:
```bash
python run.py
# Select option 3
```

- Dark theme with professional styling
- Drag & drop image upload
- Real-time analysis results
- Crown displacement calculations
- Clean, minimal design

## Batch Analysis

Process all test images:
```bash
python run.py
# Select option 2
```

Results saved to `test-image-results.csv`

## HTML Reports

Generate comprehensive reports:
```bash
python run.py
# Select option 4
```

Creates `tunnel_analysis_report.html` with:
- Embedded images
- Displacement data tables
- Professional dark theme styling
- Summary statistics

## Requirements

- Python 3.8+
- ultralytics (YOLOv11)
- opencv-python
- pandas
- pytesseract
- flask
- pillow

Install automatically via option 5 in `run.py`

## Technical Details

**Object Detection**: YOLOv11 model trained to detect:
- Tunnel crowns
- Legend labels
- Tunnel openings

**Displacement Analysis**: 
- OCR extraction of legend values
- Color-based displacement matching
- Statistical analysis per crown

**Image Processing**:
- Automatic image preprocessing
- Color space analysis
- Edge detection for crown isolation

## Data Format

**Training Images**: Located in `train-images/` with YOLO format annotations
**Test Images**: Sequential naming `test-001.png` through `test-026.png`
**Results**: CSV output with crown ID, displacement, and confidence scores

## License

Open source - extend and modify as needed for your RS2 analysis workflows.
