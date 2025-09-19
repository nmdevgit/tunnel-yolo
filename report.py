#!/usr/bin/env python3
import os
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
import base64
from ultralytics import YOLO
from datetime import datetime

def extract_legend(image_crop):
    scale_percent = 200
    w = int(image_crop.shape[1] * scale_percent / 100)
    h = int(image_crop.shape[0] * scale_percent / 100)
    resized = cv2.resize(image_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config='--oem 3 --psm 6 outputbase digits')
    
    displacements = []
    for line in text.splitlines():
        line = line.strip().replace('-', 'e-') if re.match(r'^\d+\.\d+-\d+$', line) else line
        try:
            val = float(line)
            displacements.append(val)
        except:
            continue
    
    return displacements[1:-1] if len(displacements) >= 3 else []

def extract_colors(resized_legend, num_bands):
    color_bar_area = resized_legend[50:-50, 20:60]
    band_height = color_bar_area.shape[0] // num_bands
    
    colors = []
    for i in range(num_bands):
        y1, y2 = i * band_height, (i + 1) * band_height
        band = color_bar_area[y1:y2, :]
        mean_color = tuple(int(c) for c in cv2.mean(band)[:3])
        colors.append(mean_color)
    
    return colors[1:-1] if len(colors) >= 3 else colors

def color_distance(c1, c2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))

def analyze_crown_color(crown_crop):
    edges = cv2.Canny(crown_crop, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    mask = cv2.bitwise_not(dilated)
    
    bg_pixels = crown_crop[mask == 255]
    if len(bg_pixels) == 0:
        return None
    
    return tuple(int(c) for c in np.mean(bg_pixels, axis=0))

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def generate_report():
    if not os.path.exists('runs/detect/train/weights/best.pt'):
        print("No trained model found. Run training first.")
        return
    
    print("Loading YOLO model...")
    model = YOLO('runs/detect/train/weights/best.pt')
    
    if not os.path.exists('test-images'):
        print("No test-images directory found.")
        return
    
    print("Analyzing images for report...")
    image_files = [f for f in os.listdir('test-images') if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    if not image_files:
        print("No images found in test-images directory.")
        return
    
    report_data = []
    
    for img_file in image_files:
        image_path = os.path.join('test-images', img_file)
        print(f"Processing {img_file}...")
        
        # Run YOLO prediction
        results = model.predict(source=image_path, save=False, verbose=False)
        result = results[0]
        image = cv2.imread(image_path)
        
        # Convert image to base64 for HTML embedding
        img_base64 = image_to_base64(image_path)
        
        # Analyze detections
        crown_results = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            
            legend_crop = None
            crowns_found = []
            
            for box, cls_id, score in zip(boxes, classes, scores):
                class_name = model.names[int(cls_id)]
                if class_name == 'label':
                    x1, y1, x2, y2 = map(int, box)
                    legend_crop = image[y1:y2, x1:x2]
                elif class_name == 'crown':
                    crowns_found.append((box, score))
            
            # Displacement analysis
            if legend_crop is not None and len(crowns_found) > 0:
                displacements = extract_legend(legend_crop)
                if displacements:
                    resized_legend = cv2.resize(legend_crop, (0, 0), fx=2, fy=2)
                    colors = extract_colors(resized_legend, len(displacements))
                    
                    df_legend = pd.DataFrame([
                        {"Displacement": d, "R": r, "G": g, "B": b}
                        for d, (r, g, b) in zip(displacements, colors)
                    ])
                    
                    for crown_idx, (crown_box, score) in enumerate(crowns_found):
                        x1, y1, x2, y2 = map(int, crown_box)
                        crown_crop = image[y1:y2, x1:x2]
                        
                        avg_color = analyze_crown_color(crown_crop)
                        if avg_color is not None:
                            df_legend["Distance"] = df_legend.apply(
                                lambda row: color_distance(avg_color, (row["R"], row["G"], row["B"])), axis=1
                            )
                            closest = df_legend.loc[df_legend["Distance"].idxmin()]
                            
                            crown_results.append({
                                'crown_id': crown_idx + 1,
                                'displacement': closest['Displacement'],
                                'confidence': score
                            })
        
        report_data.append({
            'filename': img_file,
            'image_base64': img_base64,
            'crown_results': crown_results
        })
    
    # Generate HTML report
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Tunnel Analysis Report</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #0d1117; color: #e6edf3; }}
        .container {{ max-width: 900px; margin: 40px auto; background: #161b22; padding: 24px; border-radius: 8px; border: 1px solid #30363d; }}
        h1 {{ color: #f0f6fc; text-align: center; font-weight: 600; }}
        .image-section {{ margin: 24px 0; padding: 16px; border: 1px solid #30363d; border-radius: 6px; background: #0d1117; }}
        .image-title {{ font-size: 16px; font-weight: 500; color: #f0f6fc; margin-bottom: 8px; }}
        .image-container {{ text-align: center; margin: 16px 0; }}
        .tunnel-image {{ max-width: 400px; height: auto; border: 1px solid #30363d; border-radius: 4px; }}
        .results-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        .results-table th, .results-table td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }}
        .results-table th {{ background: #21262d; color: #f0f6fc; font-weight: 500; }}
        .results-table td {{ color: #e6edf3; }}
        .no-crowns {{ color: #7d8590; font-style: italic; }}
        .summary {{ background: #0d1117; padding: 16px; border-radius: 4px; margin: 16px 0; border: 1px solid #30363d; }}
        .timestamp {{ text-align: center; color: #7d8590; margin-top: 20px; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Tunnel Analysis Report</h1>
        <div class="summary">
            <strong>Analysis Summary</strong><br>
            Images analyzed: {len(report_data)}<br>
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}<br>
            Total crowns found: {sum(len(item['crown_results']) for item in report_data)}
        </div>
"""
    
    for item in report_data:
        html_content += f"""
        <div class="image-section">
            <div class="image-title">{item['filename']}</div>
            <div class="image-container">
                <img src="data:image/png;base64,{item['image_base64']}" class="tunnel-image" alt="{item['filename']}">
            </div>
"""
        
        if item['crown_results']:
            html_content += """
            <table class="results-table">
                <tr>
                    <th>Crown ID</th>
                    <th>Displacement</th>
                    <th>Detection Confidence</th>
                </tr>
"""
            for crown in item['crown_results']:
                html_content += f"""
                <tr>
                    <td>Crown {crown['crown_id']}</td>
                    <td>{crown['displacement']:.4f}</td>
                    <td>{crown['confidence']:.3f}</td>
                </tr>
"""
            html_content += "</table>"
        else:
            html_content += '<div class="no-crowns">No crowns detected or analyzed</div>'
        
        html_content += "</div>"
    
    html_content += """
        <div class="timestamp">
            Report generated by Tunnel YOLO System
        </div>
    </div>
</body>
</html>
"""
    
    # Save report
    report_file = "tunnel_analysis_report.html"
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated: {report_file}")
    print(f"Open in browser to view the complete analysis report")

if __name__ == "__main__":
    generate_report()