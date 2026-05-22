import cv2
import numpy as np
import os
import glob
from src.calibration.rink_calibrator import RinkCalibrator

# Set up paths
input_dir = 'annotation_frames'
output_dir = 'tmp_frames'
os.makedirs(output_dir, exist_ok=True)

files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
calibrator = RinkCalibrator()

print(f"{'Filename':<30} | {'Mask %':<10} | {'BBox':<40}")
print("-" * 85)

for fpath in files:
    fname = os.path.basename(fpath)
    frame = cv2.imread(fpath)
    if frame is None:
        continue
    
    # Run calibration and mask extraction
    calibrator.calibrate(frame)
    mask = calibrator.get_board_mask(frame)
    
    if mask is not None and mask.any():
        coverage = (mask > 0).mean() * 100
        
        # Get bounding box
        coords = np.argwhere(mask > 0)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        bbox = f"[{x0},{y0},{x1},{y1}]"
        
        # Save overlay
        overlay = frame.copy()
        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0], dtype=np.uint8) * 0.5
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
        out_path = os.path.join(output_dir, fname.replace('.jpg', '_hybrid_latest.png'))
        cv2.imwrite(out_path, overlay)
    else:
        coverage = 0.0
        bbox = "N/A"
        
    print(f"{fname:<30} | {coverage:<10.2f} | {bbox:<40}")

