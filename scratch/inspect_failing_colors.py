#!/usr/bin/env python3
import cv2
import numpy as np
import os

failing_frames = [
    '2026-05-19_23-23-08_f0660.jpg',
    '2026-05-19_23-23-35_f0120.jpg',
    '2026-05-19_23-32-57_f0400.jpg',
    '2026-05-19_23-34-03_f0140.jpg',
    '2026-05-19_23-34-03_f0280.jpg'
]

input_dir = 'annotation_frames/new_batch/annotated'

for base in failing_frames:
    frame_path = os.path.join(input_dir, base)
    if not os.path.exists(frame_path):
        continue
    
    img = cv2.imread(frame_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print(f"\n=== Color analysis for {base} ===")
    
    # Try different blue hues
    for b_hue_min in [70, 80, 85, 90]:
        for b_sat_min in [30, 40, 50]:
            mask = cv2.inRange(hsv, np.array([b_hue_min, b_sat_min, 30]), np.array([140, 255, 255]))
            cnt = np.sum(mask > 0)
            if cnt > 100:
                print(f"  Blue [H:{b_hue_min}-140, S:{b_sat_min}-255]: {cnt} pixels")
                
    # Try different red hues
    for r_hue_max in [10, 12, 15, 20]:
        for r_sat_min in [30, 40, 50]:
            mask1 = cv2.inRange(hsv, np.array([0, r_sat_min, 30]), np.array([r_hue_max, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([180 - r_hue_max, r_sat_min, 30]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
            cnt = np.sum(mask > 0)
            if cnt > 100:
                print(f"  Red [H:0-{r_hue_max} & {180-r_hue_max}-180, S:{r_sat_min}-255]: {cnt} pixels")
