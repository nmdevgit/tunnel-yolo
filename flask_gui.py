#!/usr/bin/env python3
import os
import io
import base64
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import pandas as pd
import pytesseract
import re
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)

# Analysis functions
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

# Load model
model = None
if os.path.exists('runs/detect/train/weights/best.pt'):
    model = YOLO('runs/detect/train/weights/best.pt')

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Tunnel YOLO Analysis</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
            body { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; background: #0d1117; color: #e6edf3; }
            .container { max-width: 800px; margin: 40px auto; background: #161b22; padding: 24px; border-radius: 8px; border: 1px solid #30363d; }
            h1 { color: #f0f6fc; text-align: center; margin-bottom: 24px; font-weight: 600; }
            .upload-area { border: 2px dashed #30363d; padding: 24px; text-align: center; margin: 20px 0; border-radius: 6px; background: #0d1117; }
            .upload-area:hover { background: #21262d; border-color: #484f58; }
            input[type="file"] { margin: 8px; background: #21262d; color: #e6edf3; border: 1px solid #30363d; padding: 8px; border-radius: 4px; }
            button { background: #238636; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; font-weight: 500; }
            button:hover { background: #2ea043; }
            .results { margin-top: 16px; padding: 16px; background: #0d1117; border-radius: 4px; border: 1px solid #30363d; }
            .image-preview { max-width: 300px; margin: 16px 0; border-radius: 4px; border: 1px solid #30363d; }
            table { width: 100%; border-collapse: collapse; margin: 16px 0; }
            th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }
            th { background: #21262d; color: #f0f6fc; font-weight: 500; }
            td { color: #e6edf3; }
            .loading { display: none; text-align: center; margin: 16px; color: #7d8590; }
            .error { color: #f85149; }
            .success { color: #3fb950; }
            h3 { color: #f0f6fc; font-weight: 500; }
            h4 { color: #f0f6fc; font-weight: 500; margin: 16px 0 8px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tunnel YOLO Analysis</h1>
            
            <div class="upload-area">
                <h3>Upload Image</h3>
                <input type="file" id="imageInput" accept="image/*" onchange="previewImage()">
                <br><br>
                <button onclick="analyzeImage()">Analyze</button>
            </div>
            
            <div id="imagePreview"></div>
            <div class="loading" id="loading">Analyzing...</div>
            <div id="results"></div>
        </div>

        <script>
            function previewImage() {
                const input = document.getElementById('imageInput');
                const preview = document.getElementById('imagePreview');
                
                if (input.files && input.files[0]) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.innerHTML = '<h4>Selected Image</h4><img src="' + e.target.result + '" class="image-preview">';
                    }
                    reader.readAsDataURL(input.files[0]);
                }
            }

            function analyzeImage() {
                const input = document.getElementById('imageInput');
                const loading = document.getElementById('loading');
                const results = document.getElementById('results');
                
                if (!input.files || !input.files[0]) {
                    alert('Please select an image first');
                    return;
                }
                
                loading.style.display = 'block';
                results.innerHTML = '';
                
                const formData = new FormData();
                formData.append('image', input.files[0]);
                
                fetch('/analyze', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    loading.style.display = 'none';
                    displayResults(data);
                })
                .catch(error => {
                    loading.style.display = 'none';
                    results.innerHTML = '<div class="error">Error: ' + error + '</div>';
                });
            }

            function displayResults(data) {
                const results = document.getElementById('results');
                
                if (data.error) {
                    results.innerHTML = '<div class="error">' + data.error + '</div>';
                    return;
                }
                
                let html = '<div class="results">';
                html += '<h4 class="success">Analysis Complete</h4>';
                html += '<p>Objects found: ' + data.detections.length + '</p>';
                
                if (data.detections.length > 0) {
                    html += '<h4>Detections</h4>';
                    html += '<table><tr><th>Type</th><th>Confidence</th></tr>';
                    data.detections.forEach(det => {
                        html += '<tr><td>' + det.type + '</td><td>' + det.confidence + '</td></tr>';
                    });
                    html += '</table>';
                }
                
                if (data.crown_results && data.crown_results.length > 0) {
                    html += '<h4>Crown Displacement Analysis</h4>';
                    html += '<table><tr><th>Crown</th><th>Displacement</th><th>Color (RGB)</th></tr>';
                    data.crown_results.forEach(crown => {
                        html += '<tr><td>' + crown.crown + '</td><td>' + crown.displacement + '</td><td>' + crown.color + '</td></tr>';
                    });
                    html += '</table>';
                }
                
                html += '</div>';
                results.innerHTML = html;
            }
        </script>
    </body>
    </html>
    '''

@app.route('/analyze', methods=['POST'])
def analyze():
    if not model:
        return jsonify({'error': 'No trained model found. Run training first.'})
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    try:
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run YOLO prediction
        results = model.predict(source=img_cv2, save=False, verbose=False)
        result = results[0]
        
        if result.boxes is None:
            return jsonify({'detections': [], 'message': 'No objects detected'})
        
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        
        detections = []
        legend_crop = None
        crowns_found = []
        
        for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
            class_name = model.names[int(cls_id)]
            detections.append({
                'type': class_name,
                'confidence': f'{score:.3f}'
            })
            
            if class_name == 'label':
                x1, y1, x2, y2 = map(int, box)
                legend_crop = img_cv2[y1:y2, x1:x2]
            elif class_name == 'crown':
                crowns_found.append((box, i))
        
        crown_results = []
        
        if legend_crop is not None and len(crowns_found) > 0:
            displacements = extract_legend(legend_crop)
            if displacements:
                resized_legend = cv2.resize(legend_crop, (0, 0), fx=2, fy=2)
                colors = extract_colors(resized_legend, len(displacements))
                
                df_legend = pd.DataFrame([
                    {"Displacement": d, "R": r, "G": g, "B": b}
                    for d, (r, g, b) in zip(displacements, colors)
                ])
                
                for crown_idx, (crown_box, _) in enumerate(crowns_found):
                    x1, y1, x2, y2 = map(int, crown_box)
                    crown_crop = img_cv2[y1:y2, x1:x2]
                    
                    avg_color = analyze_crown_color(crown_crop)
                    if avg_color is not None:
                        df_legend["Distance"] = df_legend.apply(
                            lambda row: color_distance(avg_color, (row["R"], row["G"], row["B"])), axis=1
                        )
                        closest = df_legend.loc[df_legend["Distance"].idxmin()]
                        
                        crown_results.append({
                            'crown': f'Crown {crown_idx + 1}',
                            'displacement': f'{closest["Displacement"]:.4f}',
                            'color': f'({avg_color[0]}, {avg_color[1]}, {avg_color[2]})'
                        })
        
        return jsonify({
            'detections': detections,
            'crown_results': crown_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=8501)