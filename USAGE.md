# Tunnel YOLO Usage

## Quick Start

```bash
python3 run.py
```

Select from menu:
1. **Train model** - Train YOLO on tunnel data
2. **Test model** - Analyze images for displacement
3. **Install dependencies** - Install required packages

## Manual Usage

### Training
```bash
python3 train.py
```
- Uses data in `train-images/` folder
- Saves model to `runs/detect/train/weights/best.pt`
- Takes ~30-60 minutes depending on hardware

### Testing
```bash
python3 test.py
```
- Analyzes images in `test-images/` folder
- Requires trained model
- Outputs `test-image-results.csv`

## Files

- `run.py` - Main interface
- `train.py` - Training script
- `test.py` - Testing/analysis script
- `train-images/` - Training dataset
- `test-images/` - Test images
- `runs/` - Training outputs

## Output

Testing creates:
- `test-image-results.csv` - Crown displacement analysis results
- Console output with displacement values per crown

## Requirements

- Python 3.7+
- ~2GB RAM for training
- Dependencies auto-installed via run.py