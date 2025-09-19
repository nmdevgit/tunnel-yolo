#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
import pytesseract
import os
import re
from ultralytics import YOLO

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

def main():
    model = YOLO('runs/detect/train/weights/best.pt')
    results = model.predict(source='test-images', save=False)
    
    all_crown_matches = []
    
    for result in results:
        image_path = result.path
        image_name = os.path.basename(image_path).split('.')[0]
        image = cv2.imread(image_path)
        
        if result.boxes is None:
            continue
            
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        legend_crop = None
        for box, cls_id in zip(boxes, classes):
            if model.names[int(cls_id)] == 'label':
                x1, y1, x2, y2 = map(int, box)
                legend_crop = image[y1:y2, x1:x2]
                break
        
        if legend_crop is None:
            continue
        
        displacements = extract_legend(legend_crop)
        if not displacements:
            continue
        
        resized_legend = cv2.resize(legend_crop, (0, 0), fx=2, fy=2)
        colors = extract_colors(resized_legend, len(displacements))
        
        df_legend = pd.DataFrame([
            {"Displacement": d, "R": r, "G": g, "B": b}
            for d, (r, g, b) in zip(displacements, colors)
        ])
        
        crown_index = 0
        for box, cls_id in zip(boxes, classes):
            if model.names[int(cls_id)] == 'crown':
                x1, y1, x2, y2 = map(int, box)
                crown_crop = image[y1:y2, x1:x2]
                
                avg_color = analyze_crown_color(crown_crop)
                if avg_color is None:
                    continue
                
                df_legend["Distance"] = df_legend.apply(
                    lambda row: color_distance(avg_color, (row["R"], row["G"], row["B"])), axis=1
                )
                closest = df_legend.loc[df_legend["Distance"].idxmin()]
                
                all_crown_matches.append({
                    "Image": image_name,
                    "Crown_ID": crown_index,
                    "Avg_R": avg_color[0],
                    "Avg_G": avg_color[1],
                    "Avg_B": avg_color[2],
                    "Matched_Displacement": closest["Displacement"]
                })
                
                crown_index += 1
    
    df_summary = pd.DataFrame(all_crown_matches)
    df_summary.to_csv("test-image-results.csv", index=False)
    
    print("Results saved to test-image-results.csv")
    
    for _, row in df_summary.iterrows():
        print(f"{row['Image']} Crown {int(row['Crown_ID'])}: {row['Matched_Displacement']:.3f}")

if __name__ == "__main__":
    main()