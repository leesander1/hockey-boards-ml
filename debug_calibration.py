"""
debug_calibration.py - Shows color masks and projected board polygon per frame.
python debug_calibration.py --source "src/2026-05-12 21-12-18.mp4"
"""
import argparse
import cv2
import numpy as np
from src.calibration.rink_calibrator import RinkCalibrator

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
args = parser.parse_args()

cap = cv2.VideoCapture(args.source)
cal = RinkCalibrator()

for i in range(15):
    ret, frame = cap.read()
    if not ret:
        break

    red_mask  = cal._color_mask(frame, "red")
    blue_mask = cal._color_mask(frame, "blue")

    # Column sums (shows vertical stripe strength)
    red_col  = red_mask.sum(axis=0)
    blue_col = blue_mask.sum(axis=0)
    # Row sums (shows horizontal band strength)
    red_row  = red_mask.sum(axis=1)
    blue_row = blue_mask.sum(axis=1)

    print(f"Frame {i+1:02d} | "
          f"red_col_max={red_col.max():.0f} red_row_max={red_row.max():.0f} | "
          f"blue_col_max={blue_col.max():.0f} blue_row_max={blue_row.max():.0f}")

    # Save colour-mask overlays on frame 8
    if i == 7:
        dbg = frame.copy()
        dbg[red_mask  > 0] = (0,   0, 255)
        dbg[blue_mask > 0] = (255, 0,   0)
        cv2.imwrite("debug_color_masks.jpg", dbg)
        print("  → saved debug_color_masks.jpg")

cap.release()
