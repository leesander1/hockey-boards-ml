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
output_dir = 'output/debug_failures'
os.makedirs(output_dir, exist_ok=True)

for base in failing_frames:
    frame_path = os.path.join(input_dir, base)
    if not os.path.exists(frame_path):
        continue
    
    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Masks
    blue_mask = cv2.inRange(hsv, np.array([80, 40, 30]), np.array([140, 255, 255]))
    red_mask1 = cv2.inRange(hsv, np.array([0, 40, 30]), np.array([12, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([168, 40, 30]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Scorebug
    scorebug_h = int(h * 0.12)
    scorebug_w = int(w * 0.35)
    for mask in [blue_mask, red_mask]:
        mask[:scorebug_h, :scorebug_w] = 0
        mask[:int(h * 0.06), :] = 0
        mask[int(h * 0.94):, :] = 0
        mask[:, :int(w * 0.02)] = 0
        mask[:, int(w * 0.98):] = 0

    # Draw color highlights
    vis = frame.copy()
    vis[blue_mask > 0] = [255, 0, 0]  # Blue features in BGR
    vis[red_mask > 0] = [0, 0, 255]   # Red features in BGR

    # Draw a line showing scorebug exclusion
    cv2.rectangle(vis, (0, 0), (scorebug_w, scorebug_h), (0, 255, 255), 2)
    
    # Save the highlighted visualization
    out_path = os.path.join(output_dir, base)
    cv2.imwrite(out_path, vis)
    print(f"Saved visualization to {out_path}")
