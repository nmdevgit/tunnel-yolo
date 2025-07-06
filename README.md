## Tunnel-YOLO: Automating Image-Based Analysis from RS2

This repository contains a Colab-based workflow for training and deploying a YOLOv11 model to detect tunnel features—specifically tunnel openings, crowns, and embedded legend labels—from RS2-generated model outputs. It also includes a workflow for interpreting predictions to extract displacement or other metric data from detected legends using pixel-based color matching.

### Features

- Train YOLOv11 on tunnel-related classes using your own dataset
- Detect objects (tunnel-opening, crown, label) from RS2 images
- Automatically extract and match legend values from color bars
- Summarize predicted values per crown region
- Simple interface for uploading images for predictions
- Ideal for automation in numerical modelling reviews

### File Structure

- `Train_YOLO_Models_v1.ipynb`: Main Colab notebook (training + testing)
- `data-4.zip`: Training images and annotations (drag into Colab)
- `matching-images.zip`: Images for prediction/testing (drag into Colab)
- `full_legend_summary.xlsx`: Sheet-based lookup table for legend color-to-metric mapping
- `yolo11s.pt`: Pretrained YOLOv11 weights (optional)

### How to Use

1. **Open in Colab**: Clone this repo or open the notebook directly in Colab.
2. **Upload Required Files**: Drag and drop the following into the file pane:
    - `data-4.zip`
    - `matching-images.zip`
    - `full_legend_summary.xlsx`
3. **Run the Notebook**:
    - It will unzip the datasets
    - Train the model (if not already trained)
    - Detect features and generate displacement predictions
    - Display results clearly in the terminal

4. **Upload RS2 Image for Quick Testing** (Optional):
    - You can upload an image via Colab UI for single image detection and prediction summary.

### Notes

- Please ensure all RS2 legend values are in decimal format (not scientific notation).
- Model works for displacement, stress, or any other visual metric since detection is pixel-driven.
- This system is designed to be easily extended to include more features like sidewall detection and 3D section comparisons.

### Background (YOLO)

YOLO (You Only Look Once) is a real-time object detection algorithm. It processes the entire image in a single pass, predicting bounding boxes and class labels efficiently. We use it here to locate and classify components of RS2 outputs for downstream automation.

### Feedback

Please raise an issue or contact Nick if you encounter bugs or have suggestions for improvement. Contributions are welcome as this tool evolves with project needs.
