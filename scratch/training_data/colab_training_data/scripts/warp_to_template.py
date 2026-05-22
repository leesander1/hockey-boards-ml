#!/usr/bin/env python3
"""Warp annotated training frames and masks to canonical rink template space.

This script:
1. Iterates over annotated frames and their ground-truth masks.
2. Estimates a homography (world → image) using rink features or manual correspondences.
3. Computes an image → template warping matrix.
4. Warps frames and masks to a canonical template coordinate system.
5. Saves warped images and masks to disk for template-space training.

The template space represents a canonical rink view (e.g., 200 ft × 85 ft at 10 px/ft = 2000×850 px).
Training in this space removes camera viewpoint variance, helping the model generalize better.
"""

import cv2
import numpy as np
import os
import glob
import argparse
import itertools
from pathlib import Path

from src.calibration.rink_template import CALIBRATION_WORLD_POINTS
from src.calibration.rink_feature_detector import RinkFeatureDetector


# Template space constants
TEMPLATE_WIDTH = 2000  # pixels; ~200 ft at 10 px/ft
TEMPLATE_HEIGHT = 850   # pixels; ~85 ft at 10 px/ft
PIXELS_PER_WORLD_FT = 10.0


def estimate_homography_from_frame(frame: np.ndarray) -> np.ndarray | None:
    """Estimate H_world->image using robust dual-color rink line detectors.
    
    Returns the homography matrix, or None if estimation fails.
    """
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

    # Verify and construct Homography
    if best_matching:
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
        
        H, status = cv2.findHomography(world_pts, image_pts, cv2.RANSAC, 5.0)
        return H
    return None


def build_warp_matrix(H_world_to_image: np.ndarray) -> np.ndarray:
    """Build the transformation matrix from image → template space.
    
    Given H_world->image, compute H_image->template by:
    1. Invert H_world->image to get H_image->world.
    2. Compose with a pixel-scale transform that maps world coords to template pixels.
    """
    H_image_to_world = np.linalg.inv(H_world_to_image)
    
    # Pixel-scale matrix: world [ft] → template [px]
    # Template spans 200 ft × 85 ft at 10 px/ft
    T_world_to_template = np.array([
        [PIXELS_PER_WORLD_FT, 0, 0],
        [0, PIXELS_PER_WORLD_FT, 0],
        [0, 0, 1]
    ], dtype=np.float32)
    
    H_image_to_template = T_world_to_template @ H_image_to_world
    return H_image_to_template


def warp_frame_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    H_image_to_template: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Warp image and mask to template space.
    
    Returns (warped_image, warped_mask) or (None, None) on error.
    """
    try:
        warped_img = cv2.warpPerspective(
            image, H_image_to_template,
            (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        warped_msk = cv2.warpPerspective(
            mask, H_image_to_template,
            (TEMPLATE_WIDTH, TEMPLATE_HEIGHT),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return warped_img, warped_msk
    except Exception as e:
        print(f"Error warping: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description="Warp annotated frames to template space for training."
    )
    parser.add_argument(
        '--input-dir',
        default='annotation_frames/new_batch/annotated',
        help='Directory containing annotated JPG frames.'
    )
    parser.add_argument(
        '--mask-dir',
        default='test_images/annotation_frames',
        help='Directory containing corresponding PNG masks.'
    )
    parser.add_argument(
        '--output-dir',
        default='data/warped/train',
        help='Output directory for warped images and masks.'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Max frames to warp (for testing).'
    )
    args = parser.parse_args()
    
    # Create output directories
    img_out = os.path.join(args.output_dir, 'images')
    msk_out = os.path.join(args.output_dir, 'masks')
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(msk_out, exist_ok=True)
    
    # Find annotated frames
    frames = sorted(glob.glob(os.path.join(args.input_dir, '*.jpg')))
    if args.max_frames:
        frames = frames[:args.max_frames]
    
    print(f"Found {len(frames)} annotated frames. Processing...")
    
    success_count = 0
    for idx, frame_path in enumerate(frames):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[{idx+1}/{len(frames)}] SKIP: failed to read {frame_path}")
            continue
        
        # Find corresponding mask
        base = os.path.basename(frame_path)
        base_no_ext = os.path.splitext(base)[0]
        # Try both naming conventions: base-mask.png (preferred) and base.png
        mask_path = os.path.join(args.mask_dir, base_no_ext + '-mask.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(args.mask_dir, base_no_ext + '.png')
        
        if not os.path.exists(mask_path):
            print(f"[{idx+1}/{len(frames)}] SKIP: no mask for {frame_path}")
            continue
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[{idx+1}/{len(frames)}] SKIP: failed to read mask {mask_path}")
            continue
        
        # Estimate homography
        H_world_to_image = estimate_homography_from_frame(frame)
        if H_world_to_image is None:
            print(f"[{idx+1}/{len(frames)}] SKIP: failed to estimate homography for {base}")
            continue
        
        # Build warp matrix
        H_image_to_template = build_warp_matrix(H_world_to_image)
        
        # Warp
        warped_img, warped_msk = warp_frame_and_mask(frame, mask, H_image_to_template)
        if warped_img is None:
            print(f"[{idx+1}/{len(frames)}] SKIP: failed to warp {base}")
            continue
        
        # Save
        img_out_path = os.path.join(img_out, base)
        msk_out_path = os.path.join(msk_out, base_no_ext + '.png')
        cv2.imwrite(img_out_path, warped_img)
        cv2.imwrite(msk_out_path, warped_msk)
        
        success_count += 1
        print(f"[{idx+1}/{len(frames)}] OK: {base}")
    
    print(f"\nWarped {success_count}/{len(frames)} frames to {args.output_dir}")


if __name__ == '__main__':
    main()
