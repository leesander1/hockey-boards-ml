#!/usr/bin/env python3
"""Fit board height pixel model from hand-drawn annotation frames.

Produces `annotation_calibration.json` next to the rink calibrators, which
`TrimDetector` will read at runtime to override the default top/bottom px.

Usage: python3 scripts/fit_board_height_from_annotations.py
"""
import json
import os
from glob import glob
import cv2
import numpy as np


ANN_DIR = os.path.join(os.path.dirname(__file__), '..', 'annotation_frames')
OUT_FILE = os.path.join(os.path.dirname(__file__), '..', 'src', 'calibration', 'annotation_calibration.json')


def load_annotation_masks(path):
    img = cv2.imread(path)
    if img is None:
        return None, None
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Yellow annotation (trim) - loose bounds
    ylo = np.array([15, 120, 120], dtype=np.uint8)
    yhi = np.array([40, 255, 255], dtype=np.uint8)
    yellow = cv2.inRange(hsv, ylo, yhi)

    # Red annotation (top of board) - capture both hue wraps
    r1lo = np.array([0, 120, 120], dtype=np.uint8)
    r1hi = np.array([12, 255, 255], dtype=np.uint8)
    r2lo = np.array([170, 120, 120], dtype=np.uint8)
    r2hi = np.array([180, 255, 255], dtype=np.uint8)
    red = cv2.bitwise_or(cv2.inRange(hsv, r1lo, r1hi), cv2.inRange(hsv, r2lo, r2hi))

    # Clean small specks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k)
    red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k)

    return yellow, red


def extract_column_pairs(yellow_mask, red_mask):
    h, w = yellow_mask.shape[:2]
    t_cols = []
    bh = []
    for x in range(w):
        col_y = np.where(yellow_mask[:, x] > 0)[0]
        col_r = np.where(red_mask[:, x] > 0)[0]
        if col_y.size == 0 or col_r.size == 0:
            continue
        bot = int(col_y.max())
        top = int(col_r.min())
        # sanity
        board_h = bot - top
        if bot <= top or board_h <= 2:
            continue
        # The board face in these broadcast shots is never hundreds of pixels
        # tall; clip obvious color false-positives before fitting.
        if board_h < 35 or board_h > 180:
            continue
        t_cols.append(bot / float(h))
        bh.append(board_h)
    return t_cols, bh


def main():
    files = sorted(glob(os.path.join(ANN_DIR, '*.png')))
    all_t = []
    all_h = []
    for p in files:
        yellow, red = load_annotation_masks(p)
        if yellow is None:
            continue
        t_cols, bh = extract_column_pairs(yellow, red)
        if len(t_cols) == 0:
            print('no pairs in', p)
            continue
        all_t.extend(t_cols)
        all_h.extend(bh)

    if len(all_t) < 10:
        print('not enough annotation data found')
        return

    all_t = np.array(all_t, dtype=np.float32)
    all_h = np.array(all_h, dtype=np.float32)

    # Use the farthest and nearest perspective bands separately.
    # This is much more stable than a global line fit because the annotations
    # contain real broadcast colors, ads, and crowd content that create outliers.
    low_t = np.quantile(all_t, 0.15)
    high_t = np.quantile(all_t, 0.85)
    low_band = all_h[all_t <= low_t]
    high_band = all_h[all_t >= high_t]

    top_px = float(np.median(low_band)) if low_band.size else 60.0
    bottom_px = float(np.median(high_band)) if high_band.size else 80.0

    # Keep the result in a physically plausible range even if the sample set
    # is noisy.
    top_px = float(np.clip(top_px, 35.0, 90.0))
    bottom_px = float(np.clip(bottom_px, top_px + 10.0, 140.0))

    out = {
        'board_height_top_px': top_px,
        'board_height_bottom_px': bottom_px,
        'slope': float(bottom_px - top_px),
        'samples': int(all_t.size)
    }

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        json.dump(out, f, indent=2)

    print('Wrote', OUT_FILE)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
