#!/usr/bin/env python3
import cv2
import numpy as np
import os

base = '2026-05-19_23-23-08_f0660.jpg'
input_dir = 'annotation_frames/new_batch/annotated'
frame_path = os.path.join(input_dir, base)

img = cv2.imread(frame_path)
h, w = img.shape[:2]
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

blue_mask = cv2.inRange(hsv, np.array([80, 40, 30]), np.array([140, 255, 255]))

scorebug_h = int(h * 0.12)
scorebug_w = int(w * 0.35)
blue_mask[:scorebug_h, :scorebug_w] = 0
blue_mask[:int(h * 0.06), :] = 0
blue_mask[int(h * 0.94):, :] = 0
blue_mask[:, :int(w * 0.02)] = 0
blue_mask[:, int(w * 0.98):] = 0

col_sums = np.sum(blue_mask > 0, axis=0)
active_cols = [(col, col_sums[col]) for col in range(w) if col_sums[col] > 0]
print(f"Number of columns with any blue pixels: {len(active_cols)}")
print("Columns with peak sums > 5:")
# Group contiguous columns and find peaks
sorted_by_val = sorted(active_cols, key=lambda x: x[1], reverse=True)
for col, val in sorted_by_val[:20]:
    print(f"  Col {col}: sum = {val}")
