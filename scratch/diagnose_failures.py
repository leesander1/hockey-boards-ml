#!/usr/bin/env python3
import cv2
import numpy as np
import os
import glob
import sys
import itertools

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
        print(f"Skipping {base} - not found")
        continue
    
    frame = cv2.imread(frame_path)
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 1. Color Segmentation
    blue_mask = cv2.inRange(
        hsv,
        np.array([80, 40, 30], dtype=np.uint8),
        np.array([140, 255, 255], dtype=np.uint8)
    )

    red_mask1 = cv2.inRange(
        hsv,
        np.array([0, 40, 30], dtype=np.uint8),
        np.array([12, 255, 255], dtype=np.uint8)
    )
    red_mask2 = cv2.inRange(
        hsv,
        np.array([168, 40, 30], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8)
    )
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Crop/mask out scorebug & extreme borders
    scorebug_h = int(h * 0.12)
    scorebug_w = int(w * 0.35)
    for mask in [blue_mask, red_mask]:
        mask[:scorebug_h, :scorebug_w] = 0
        mask[:int(h * 0.06), :] = 0
        mask[int(h * 0.94):, :] = 0
        mask[:, :int(w * 0.02)] = 0
        mask[:, int(w * 0.98):] = 0

    # Peak detection
    def find_peaks(mask, min_density=25, min_dist=80):
        col_sums = np.sum(mask > 0, axis=0)
        peaks = []
        for col in range(w):
            val = col_sums[col]
            if val < min_density:
                continue
            left_bound = max(0, col - min_dist)
            right_bound = min(w, col + min_dist + 1)
            if val == np.max(col_sums[left_bound:right_bound]):
                duplicates = np.where(col_sums[left_bound:right_bound] == val)[0] + left_bound
                mean_col = int(np.mean(duplicates))
                if mean_col not in [p[0] for p in peaks]:
                    peaks.append((mean_col, val))
        
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        filtered_peaks = []
        for p in peaks:
            col, val = p
            too_close = False
            for fp in filtered_peaks:
                if abs(fp[0] - col) < min_dist:
                    too_close = True
                    break
            if not too_close:
                filtered_peaks.append((col, val))
        return sorted([fp[0] for fp in filtered_peaks])

    blue_peaks = find_peaks(blue_mask, min_density=25, min_dist=80)
    red_peaks = find_peaks(red_mask, min_density=25, min_dist=80)

    # Fit lines and filter candidates
    def get_line_candidates(mask, peaks, window=25):
        candidates = []
        for peak_col in peaks:
            ys, xs = np.where(mask[:, max(0, peak_col-window):min(w, peak_col+window+1)] > 0)
            if len(ys) < 30:
                continue
            xs = xs + max(0, peak_col-window)
            y_min = int(np.min(ys))
            y_max = int(np.max(ys))
            span = y_max - y_min
            
            # Polyfit slope and fit check
            a, b = np.polyfit(ys, xs, 1)
            
            candidates.append({
                'col': peak_col,
                'a': a,
                'b': b,
                'y_min': y_min,
                'y_max': y_max,
                'span': span,
                'x_top': a * y_min + b,
                'x_bot': a * y_max + b
            })
        return candidates

    blue_candidates = get_line_candidates(blue_mask, blue_peaks)
    red_candidates = get_line_candidates(red_mask, red_peaks)

    print(f"\n--- Frame: {base} ---")
    print(f"Blue peaks: {blue_peaks}")
    print(f"Red peaks: {red_peaks}")
    print(f"Blue Candidates:")
    for bc in blue_candidates:
        print(f"  col={bc['col']}: span={bc['span']}, y_range=[{bc['y_min']}, {bc['y_max']}], slope={bc['a']:.3f}, ok_span={bc['span']>=100}, ok_slope={abs(bc['a'])<=0.30}")
    print(f"Red Candidates:")
    for rc in red_candidates:
        print(f"  col={rc['col']}: span={rc['span']}, y_range=[{rc['y_min']}, {rc['y_max']}], slope={rc['a']:.3f}, ok_span={rc['span']>=100}, ok_slope={abs(rc['a'])<=0.30}")
