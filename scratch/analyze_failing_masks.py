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
    img_path = os.path.join(input_dir, base)
    mask_path = os.path.join(input_dir, os.path.splitext(base)[0] + '-mask.png')
    if not os.path.exists(mask_path):
        mask_path = os.path.join(input_dir, os.path.splitext(base)[0] + '.png')
    
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        print(f"File missing for {base}")
        continue
        
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"\n--- {base} Analysis ---")
    print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
    print(f"Mask unique values: {np.unique(mask)}")
    print(f"Mask white pixels (boards): {np.sum(mask > 0)}")
