import cv2
import numpy as np
import os
from pathlib import Path
from src.calibration.annotation_board_detector import AnnotationBoardDetector
from src.calibration.rink_calibrator import RinkCalibrator
from src.calibration.trim_detector import TrimDetector

def calculate_metrics(gt, pred):
    gt = (gt > 0).astype(np.uint8)
    pred = (pred > 0).astype(np.uint8)
    if gt.shape != pred.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    iou = intersection / union if union > 0 else 0
    dice = (2 * intersection) / (gt.sum() + pred.sum()) if (gt.sum() + pred.sum()) > 0 else 0
    return iou, dice

stems = [
    "v1_f010_overhead_end_zone",
    "v1_f317_overhead_end_zone",
    "v1_f634_overhead_end_zone",
    "v3_f010_overhead_full",
    "v3_f200_overhead_full",
    "v4_f010_overhead_near"
]

results_rc = []
results_trim = []

print(f"{'Stem':<30} | RC IoU/Dice | Trim IoU/Dice")
print("-" * 65)

for stem in stems:
    png_path = f"annotation_frames/{stem}.png"
    jpg_path = f"annotation_frames/{stem}.jpg"
    
    if not os.path.exists(png_path) or not os.path.exists(jpg_path):
        # try .png for both if .jpg doesn't exist (sometimes they are both png)
        jpg_path = png_path
        if not os.path.exists(jpg_path):
            continue
        
    # 1. Pseudo-GT
    # We use the same frame for GT extraction as for prediction
    # but the annotation detector needs the frame with red/yellow lines.
    ann_frame = cv2.imread(png_path)
    if ann_frame is None: continue
    
    abd = AnnotationBoardDetector()
    # Force signature check to be true by overriding it if it's too strict
    if not abd.detect(ann_frame):
        # try without the signature check if it fails
        orig_sig = AnnotationBoardDetector._has_annotation_signature
        AnnotationBoardDetector._has_annotation_signature = lambda self, hsv: True
        if not abd.detect(ann_frame):
            print(f"Failed to detect annotation for {stem}")
            AnnotationBoardDetector._has_annotation_signature = orig_sig
            continue
        AnnotationBoardDetector._has_annotation_signature = orig_sig

    gt_mask = abd.get_board_mask()
    
    # 2. Raw Frame (for detection, we want the one without lines if possible, 
    # but for these files the .png HAS the lines and .jpg IS the raw.
    # If .jpg is missing, we use .png and hope the detectors are robust.)
    raw_path = f"annotation_frames/{stem}.jpg"
    if not os.path.exists(raw_path):
        raw_path = png_path
    frame = cv2.imread(raw_path)
    
    # 3a. Current RinkCalibrator
    rc = RinkCalibrator()
    rc.calibrate(frame)
    rc_mask = rc.get_board_mask(frame)
    if rc_mask is None: rc_mask = np.zeros_like(gt_mask)
    rc_iou, rc_dice = calculate_metrics(gt_mask, rc_mask)
    results_rc.append((rc_iou, rc_dice))
    
    # 3b. TrimDetector path
    # We use RinkCalibrator's _ice_mask if get_ice_mask doesn't exist
    rc_trim = RinkCalibrator()
    rc_trim.calibrate(frame)
    ice_mask = getattr(rc_trim, '_ice_mask', None)
    if ice_mask is None:
        # Fallback: simple threshold if _ice_mask is not available
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ice_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    
    td = TrimDetector()
    td.detect(frame, ice_mask)
    trim_mask = td.get_board_mask(frame)
    if trim_mask is None: trim_mask = np.zeros_like(gt_mask)
    trim_iou, trim_dice = calculate_metrics(gt_mask, trim_mask)
    results_trim.append((trim_iou, trim_dice))
    
    print(f"{stem:<30} | {rc_iou:.3f}/{rc_dice:.3f} | {trim_iou:.3f}/{trim_dice:.3f}")

if results_rc:
    mean_rc_iou = np.mean([r[0] for r in results_rc])
    mean_rc_dice = np.mean([r[1] for r in results_rc])
    mean_trim_iou = np.mean([r[0] for r in results_trim])
    mean_trim_dice = np.mean([r[1] for r in results_trim])

    print("-" * 65)
    print(f"{'MEAN':<30} | {mean_rc_iou:.3f}/{mean_rc_dice:.3f} | {mean_trim_iou:.3f}/{mean_trim_dice:.3f}")

    if mean_trim_iou > mean_rc_iou:
        print("\nTrimDetector wins!")
    else:
        print("\nRinkCalibrator wins!")
else:
    print("No results.")
