#!/usr/bin/env python3
"""Test the ML board detector on annotated frames."""

import cv2
import numpy as np
import os
from glob import glob
from src.calibration.ml_board_detector import MLBoardDetector

# Wait for model to exist
MODEL_PATH = 'src/calibration/board_segmentation_model.pth'
ann_dir = 'annotation_frames'
out_dir = 'tmp_frames'
os.makedirs(out_dir, exist_ok=True)

while not os.path.exists(MODEL_PATH):
    print(f'Waiting for model at {MODEL_PATH}...')
    import time
    time.sleep(2)

print(f'Model found! Testing ML detector...')

# Load detector
detector = MLBoardDetector()

# Test on all source frames
for frame_path in sorted(glob(os.path.join(ann_dir, '*.jpg'))):
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f'Failed to load {frame_path}')
        continue
    
    ok = detector.detect(frame)
    if ok:
        mask = detector.get_board_mask()
        if mask is not None:
            ys, xs = np.where(mask > 0)
            h, w = frame.shape[:2]
            cov = (mask > 0).mean() * 100
            fname = os.path.basename(frame_path)
            print(f'{fname:35s} bbox x[{xs.min():4d},{xs.max():4d}] y[{ys.min():3d},{ys.max():3d}] cov {cov:5.1f}%')
            
            # Visualize
            out_path = os.path.join(out_dir, fname.replace('.jpg', '_ml_board_overlay.png'))
            overlay = frame.copy()
            color = np.zeros_like(frame)
            color[:, :] = (0, 0, 255)
            alpha = 0.45
            overlay = np.where(mask[:, :, None] > 0, (overlay * (1 - alpha) + color * alpha).astype(np.uint8), overlay)
            cv2.imwrite(out_path, overlay)
            print(f'  Saved overlay: {out_path}')
        else:
            print(f'{os.path.basename(frame_path):35s} mask is None')
    else:
        print(f'{os.path.basename(frame_path):35s} detection failed')

print('Done!')
