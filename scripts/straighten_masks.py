#!/usr/bin/env python3
"""
Straighten wobbly, hand-drawn board annotations using 2nd-degree polynomial curve-fitting.
Overwrites the *-mask.png files with mathematically perfect boundaries.
"""

import cv2
import numpy as np
import os
import glob
from scipy.ndimage import median_filter

ANNOTATED_DIR = 'annotation_frames/new_batch/annotated'
DEBUG_DIR = 'annotation_frames/debug_straightened'

def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    
    # Find all annotated PNGs
    annotated_paths = sorted(glob.glob(os.path.join(ANNOTATED_DIR, '*-annotated.png')) + 
                             glob.glob(os.path.join(ANNOTATED_DIR, '*-annotate.png')))
    
    print(f"Found {len(annotated_paths)} annotated frames to straighten.")
    
    straightened_count = 0
    
    for idx, ann_path in enumerate(annotated_paths):
        # 1. Determine filenames
        base_dir = os.path.dirname(ann_path)
        base_name = os.path.basename(ann_path)
        
        # Strip suffix to get original image stem
        stem = base_name.replace('-annotated.png', '').replace('-annotate.png', '')
        
        jpg_path = os.path.join(base_dir, f"{stem}.jpg")
        old_mask_path = os.path.join(base_dir, f"{stem}-mask.png")
        
        if not os.path.exists(jpg_path):
            print(f"[{idx+1}/{len(annotated_paths)}] Skip: {jpg_path} not found.")
            continue
            
        # 2. Load original image and annotated image
        img = cv2.imread(jpg_path)
        ann = cv2.imread(ann_path)
        
        if img is None or ann is None:
            print(f"[{idx+1}/{len(annotated_paths)}] Skip: failed to read images.")
            continue
            
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(ann, cv2.COLOR_BGR2HSV)
        
        # 3. Segment manual red/yellow marker lines (highly saturated, bright pen strokes)
        # Red marker (top)
        red = cv2.inRange(hsv, np.array([0, 140, 120], np.uint8), np.array([10, 255, 255], np.uint8))
        red = cv2.bitwise_or(red, cv2.inRange(hsv, np.array([170, 140, 120], np.uint8), np.array([180, 255, 255], np.uint8)))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        
        # Yellow marker (bottom)
        yellow = cv2.inRange(hsv, np.array([15, 140, 120], np.uint8), np.array([35, 255, 255], np.uint8))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)))
        
        # 4. Extract column-wise centroids
        red_top_y = np.full(w, np.nan, dtype=np.float32)
        yellow_bot_y = np.full(w, np.nan, dtype=np.float32)
        
        for col in range(w):
            r_ys = np.where(red[:, col] > 0)[0]
            y_ys = np.where(yellow[:, col] > 0)[0]
            if r_ys.size:
                red_top_y[col] = float(r_ys.min())
            if y_ys.size:
                yellow_bot_y[col] = float(y_ys.max())
                
        # Find column range where lines actually exist to cut off at the ends
        valid_red = ~np.isnan(red_top_y)
        valid_yellow = ~np.isnan(yellow_bot_y)
        valid_cols = np.where(valid_red & valid_yellow)[0]
        
        if len(valid_cols) < w * 0.08:
            print(f"[{idx+1}/{len(annotated_paths)}] Skip {stem}: insufficient annotated columns.")
            continue
            
        ann_col_lo = valid_cols.min()
        ann_col_hi = valid_cols.max()
        
        # 5. Fit 2nd-degree polynomials directly to the sparse valid coordinates
        x_coords = np.arange(w, dtype=np.float32)
        
        # Top boundary
        coeffs_top = np.polyfit(x_coords[valid_red], red_top_y[valid_red], deg=2)
        top_smooth = np.polyval(coeffs_top, x_coords)
        
        # Bottom boundary
        coeffs_bot = np.polyfit(x_coords[valid_yellow], yellow_bot_y[valid_yellow], deg=2)
        bot_smooth = np.polyval(coeffs_bot, x_coords)
        
        # 6. Construct perfectly straight/curved binary mask
        perfect_mask = np.zeros((h, w), dtype=np.uint8)
        for col in range(ann_col_lo, ann_col_hi + 1):
            t_y = int(np.clip(top_smooth[col], 0, h - 1))
            b_y = int(np.clip(bot_smooth[col], t_y + 1, h - 1))
            perfect_mask[t_y:b_y, col] = 255
            
        # Exclude scorebug zone (matching baseline dataset prep)
        perfect_mask[:int(h * 0.25), :int(w * 0.35)] = 0
        
        # 7. Save corrected binary mask to disk
        cv2.imwrite(old_mask_path, perfect_mask)
        
        # 8. Create debug side-by-side comparison
        # Read old mask if it exists
        old_mask = np.zeros_like(perfect_mask)
        if os.path.exists(old_mask_path) and straightened_count > 0: # Note: we just overwrote it, let's load or skip
            pass # We can overlay perfect mask in green and manual markings in red/blue
            
        # Overlay original image with old wobbly drawing and new perfect fit
        overlay_old = img.copy()
        # Draw raw marker boundaries in red (top) and yellow (bottom)
        for col in range(w):
            if not np.isnan(red_top_y[col]):
                cv2.circle(overlay_old, (col, int(red_top_y[col])), 2, (0, 0, 255), -1)
            if not np.isnan(yellow_bot_y[col]):
                cv2.circle(overlay_old, (col, int(yellow_bot_y[col])), 2, (0, 255, 255), -1)
                
        overlay_new = img.copy()
        # Overlay the perfect mask in semi-transparent green
        green_mask = np.zeros_like(img)
        green_mask[perfect_mask > 0] = (0, 255, 0)
        cv2.addWeighted(overlay_new, 1.0, green_mask, 0.4, 0, dst=overlay_new)
        
        # Draw the perfectly straight polynomial boundary lines
        for col in range(ann_col_lo, ann_col_hi + 1):
            cv2.circle(overlay_new, (col, int(top_smooth[col])), 1, (255, 0, 0), -1)  # Blue top line
            cv2.circle(overlay_new, (col, int(bot_smooth[col])), 1, (0, 0, 255), -1)  # Red bottom line
            
        # Add labels
        cv2.putText(overlay_old, "Original Frame + Raw Hand Drawings", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(overlay_new, "Denoised Perfectly Straight Board Mask", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Concatenate side-by-side
        comp = np.hstack((overlay_old, overlay_new))
        
        # Save comparison
        debug_out_path = os.path.join(DEBUG_DIR, f"{stem}_comparison.png")
        cv2.imwrite(debug_out_path, comp)
        
        straightened_count += 1
        print(f"[{idx+1}/{len(annotated_paths)}] OK: {stem} (columns {ann_col_lo}-{ann_col_hi})")
        
    print(f"\nSuccessfully straightened {straightened_count} annotation masks!")
    print(f"Debug comparison frames saved in: {DEBUG_DIR}")

if __name__ == "__main__":
    main()
