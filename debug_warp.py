#!/usr/bin/env python3
"""Debug script to visualize the warping issue."""

import cv2
import numpy as np
import sys
import itertools
from src.calibration.rink_template import CALIBRATION_WORLD_POINTS
from src.calibration.rink_feature_detector import RinkFeatureDetector

frame_path = 'annotation_frames/new_batch/annotated/2026-05-12_21-12-18_f0132.jpg'
frame = cv2.imread(frame_path)

print(f"Frame shape: {frame.shape}")
print(f"World points: {CALIBRATION_WORLD_POINTS}")

# Detect features using robust dual-color rink line detectors
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

# Peak detection with robust min_density and min_dist
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

print(f"\nDetected peaks:")
print(f"  Blue column peaks: {blue_peaks}")
print(f"  Red column peaks: {red_peaks}")

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
        
        # Filters
        if span < 100:
            continue
        
        a, b = np.polyfit(ys, xs, 1)
        if abs(a) > 0.30: # keep near-vertical strokes only
            continue
            
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

print(f"\nLine candidates:")
print(f"  Blue candidates: {[c['col'] for c in blue_candidates]}")
print(f"  Red candidates: {[c['col'] for c in red_candidates]}")

# Generate and evaluate all possible matchings
best_matching = None
best_score = -1

# Red choices (None or 1 candidate)
red_choices = [None] + red_candidates

# Blue choices (None, 1, or 2 candidates)
blue_matches = []
for r in range(3):
    for combo in itertools.combinations(blue_candidates, r):
        if len(combo) == 0:
            blue_matches.append((None, None))
        elif len(combo) == 1:
            blue_matches.append((combo[0], None))
            blue_matches.append((None, combo[0]))
        elif len(combo) == 2:
            c1, c2 = combo
            y_mid = int((c1['y_min'] + c1['y_max'] + c2['y_min'] + c2['y_max']) / 4)
            x1 = c1['a'] * y_mid + c1['b']
            x2 = c2['a'] * y_mid + c2['b']
            if x1 < x2:
                blue_matches.append((c1, c2))
            else:
                blue_matches.append((c2, c1))

for red_cand in red_choices:
    for left_blue_cand, right_blue_cand in blue_matches:
        matching = {}
        if red_cand is not None:
            matching[100.0] = red_cand
        if left_blue_cand is not None:
            matching[75.0] = left_blue_cand
        if right_blue_cand is not None:
            matching[125.0] = right_blue_cand

        if len(matching) < 2:
            continue

        # Enforce ordering in image space
        y_mid = int(h / 2)
        sorted_keys = sorted(matching.keys())
        x_coords = [matching[k]['a'] * y_mid + matching[k]['b'] for k in sorted_keys]
        
        ordered = True
        for i in range(len(x_coords) - 1):
            if x_coords[i] >= x_coords[i+1]:
                ordered = False
                break
        if not ordered:
            continue

        # Validate scale consistency and y_max consistency (near boards boundary)
        scales = {}
        valid_scales = True
        for i in range(len(sorted_keys)):
            for j in range(i + 1, len(sorted_keys)):
                w1, w2 = sorted_keys[i], sorted_keys[j]
                c1, c2 = matching[w1], matching[w2]
                
                # Bottom y_max must be consistent across matched lines
                if abs(c1['y_max'] - c2['y_max']) > 100:
                    valid_scales = False
                    break
                
                d_world = abs(w1 - w2)
                x1 = c1['a'] * y_mid + c1['b']
                x2 = c2['a'] * y_mid + c2['b']
                d_image = abs(x1 - x2)
                
                scale = d_image / d_world
                scales[(w1, w2)] = scale
                
                # Scale must be reasonable
                if scale < 5.0 or scale > 80.0:
                    valid_scales = False
                    break
            if not valid_scales:
                break
        
        if not valid_scales:
            continue

        # If 3 lines are matched, check scale ratio (should be close to 1.0)
        if len(matching) == 3:
            scale_left = scales[(75.0, 100.0)]
            scale_right = scales[(100.0, 125.0)]
            ratio = scale_left / scale_right
            if ratio < 0.65 or ratio > 1.55:
                continue

        # Base score: sum of spans of matched lines
        score = sum(c['span'] for c in matching.values())

        if score > best_score:
            best_score = score
            best_matching = matching

if best_matching is None:
    print("No valid matching found!")
    sys.exit(1)

print(f"\nBest Selected Matching:")
for k in sorted(best_matching.keys()):
    print(f"  World x={k} -> col={best_matching[k]['col']} (y_range=[{best_matching[k]['y_min']}, {best_matching[k]['y_max']}])")

# Construct points for Homography
world_pts = []
image_pts = []

for wx in sorted(best_matching.keys()):
    lc = best_matching[wx]
    image_pts.append([lc['x_bot'], lc['y_max']])
    world_pts.append([wx, 0.0])
    
    image_pts.append([lc['x_top'], lc['y_min']])
    world_pts.append([wx, 85.0])

world_pts = np.array(world_pts, dtype=np.float32)
image_pts = np.array(image_pts, dtype=np.float32)

print(f"\nWorld points mapping:")
for i, pt in enumerate(world_pts):
    print(f"  {i}: {pt} -> Image: {image_pts[i]}")

# Compute homography
H, status = cv2.findHomography(world_pts, image_pts, cv2.RANSAC, 5.0)
print(f"\nH_world->image:\n{H}")

# Test the homography
print(f"\nVerify homography (world -> image):")
for i, (wx, wy) in enumerate(world_pts):
    pt_world = np.array([wx, wy, 1])
    pt_image_proj = H @ pt_world
    pt_image_norm = pt_image_proj[:2] / pt_image_proj[2]
    print(f"  World {i} ({wx}, {wy}) -> Image {pt_image_norm} (expected {image_pts[i]})")

# Inverse
H_inv = np.linalg.inv(H)
print(f"\nH_image->world (inverted):\n{H_inv}")

# Pixel scale
PIXELS_PER_FT = 10.0
T = np.array([
    [PIXELS_PER_FT, 0, 0],
    [0, PIXELS_PER_FT, 0],
    [0, 0, 1]
], dtype=np.float32)

H_image_to_template = T @ H_inv
print(f"\nH_image->template (pixels):\n{H_image_to_template}")

# Test a few points
print(f"\nTest warping corners:")
template_size = (2000, 850)

corners_image = np.array([
    [0, 0],          # top-left
    [frame.shape[1]-1, 0],    # top-right
    [0, frame.shape[0]-1],    # bottom-left
    [frame.shape[1]-1, frame.shape[0]-1],  # bottom-right
], dtype=np.float32)

for i, (px, py) in enumerate(corners_image):
    pt = np.array([px, py, 1])
    pt_template = H_image_to_template @ pt
    pt_template_norm = pt_template[:2] / pt_template[2]
    print(f"  Image corner {i} ({px}, {py}) -> Template {pt_template_norm}")

print(f"\nTemplate size: {template_size}")
print(f"Template world coverage: 0-200 ft (x) × 0-85 ft (y)")
print(f"Template pixel coverage: 0-2000 px (x) × 0-850 px (y)")
