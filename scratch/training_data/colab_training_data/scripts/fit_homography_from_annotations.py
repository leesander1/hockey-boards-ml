#!/usr/bin/env python3
"""Fit a homography from world (rink) to image coordinates using annotation lines.

The annotation frames have red lines at the top of boards and yellow lines at the
bottom. We extract those and use the known rink geometry to compute a homography
that maps world coordinates to image coordinates.

Usage: python3 scripts/fit_homography_from_annotations.py
"""
import json
import os
from glob import glob
import cv2
import numpy as np

ANN_DIR = os.path.join(os.path.dirname(__file__), '..', 'annotation_frames')
OUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'src', 'calibration', 'annotation_homography.json')

# Import world points from rink template
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.calibration.rink_template import (
    CALIBRATION_WORLD_POINTS,
    BLUE_LINE_LEFT_X, BLUE_LINE_RIGHT_X, CENTER_LINE_X,
    RINK_WIDTH
)


def extract_line_centers(path):
    """
    Extract the per-column y-position of the red (top) and yellow (bottom) lines.
    Returns (red_ys, yellow_ys) where each is a (w,) array with per-column y or NaN.
    """
    img = cv2.imread(path)
    if img is None:
        return None, None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # Red annotation (top of board)
    red = cv2.inRange(hsv, np.array([0, 150, 150], np.uint8), np.array([10, 255, 255], np.uint8))
    red = cv2.bitwise_or(red, cv2.inRange(hsv, np.array([170, 150, 150], np.uint8), np.array([180, 255, 255], np.uint8)))
    red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

    # Yellow annotation (bottom of board)
    yellow = cv2.inRange(hsv, np.array([18, 150, 150], np.uint8), np.array([35, 255, 255], np.uint8))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

    red_ys = np.full(w, np.nan, dtype=np.float32)
    yellow_ys = np.full(w, np.nan, dtype=np.float32)

    for col in range(w):
        r_ys = np.where(red[:, col] > 0)[0]
        y_ys = np.where(yellow[:, col] > 0)[0]
        if r_ys.size:
            red_ys[col] = float(r_ys.min())
        if y_ys.size:
            yellow_ys[col] = float(y_ys.max())

    return red_ys, yellow_ys


def extract_correspondence_points():
    """
    Extract image-space correspondence points from annotation frames.
    Returns list of (world_point, image_point) tuples.
    """
    correspondences = []
    
    # World calibration points (from rink_template.py)
    world_pts = np.array(CALIBRATION_WORLD_POINTS, dtype=np.float32)
    
    for path in sorted(glob(os.path.join(ANN_DIR, '*.png'))):
        red_ys, yellow_ys = extract_line_centers(path)
        if red_ys is None:
            continue

        img = cv2.imread(path)
        w = img.shape[1]

        # For each world point, find where it projects to in this image
        for world_x, world_y in world_pts:
            # Estimate which column this x-coordinate maps to
            # This is a rough estimate: assume the rink is roughly centered and
            # scales linearly across the frame width
            # We'll use the lines that are actually visible and work backward
            
            # Map world x to approximate image column
            # Rough: assume the 200-ft rink spans the frame width (will refine with homography)
            img_col = int((world_x / 200.0) * w)
            img_col = max(0, min(w - 1, img_col))

            # Use the appropriate line based on world_y
            if world_y < 5.0:  # near boards
                y_vals = red_ys if world_y < 0 else yellow_ys
            else:  # far boards
                y_vals = red_ys if world_y < RINK_WIDTH * 0.5 else yellow_ys

            # Search for a strong line response near this column
            search_range = 50
            lo = max(0, img_col - search_range)
            hi = min(w, img_col + search_range)

            if world_y < 42.5:  # use yellow (bottom)
                col_vals = yellow_ys[lo:hi]
            else:  # use red (top)
                col_vals = red_ys[lo:hi]

            valid = ~np.isnan(col_vals)
            if valid.sum() > 0:
                # Take median of the valid columns near our guess
                img_y = np.median(col_vals[valid])
                best_col = lo + np.nanargmax(valid.astype(float))
                correspondences.append({
                    'world': (float(world_x), float(world_y)),
                    'image': (float(best_col), float(img_y))
                })

    return correspondences


def fit_homography_from_points(correspondences):
    """Compute a homography from correspondence points."""
    if len(correspondences) < 4:
        print(f'not enough correspondence points: {len(correspondences)}')
        return None

    world_pts = np.array([c['world'] for c in correspondences], dtype=np.float32)
    image_pts = np.array([c['image'] for c in correspondences], dtype=np.float32)

    h, mask = cv2.findHomography(world_pts, image_pts)
    if h is None:
        print('failed to compute homography')
        return None

    return h


def main():
    print('Extracting correspondence points from annotation frames...')
    correspondences = extract_correspondence_points()

    if len(correspondences) < 4:
        print(f'not enough points: {len(correspondences)}')
        return

    print(f'Found {len(correspondences)} correspondences')

    h = fit_homography_from_points(correspondences)
    if h is None:
        return

    # Serialize
    out = {
        'homography': h.tolist(),
        'num_points': len(correspondences),
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(out, f, indent=2)

    print('Wrote', OUT_FILE)
    print('Homography shape:', h.shape)
    print('Matrix:\n', h)


if __name__ == '__main__':
    main()
