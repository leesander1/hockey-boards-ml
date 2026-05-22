"""
Debug script: tests the TrimDetector against the annotation reference frames.
Outputs three images per frame:
  - *_trim.jpg   : the raw yellow-trim colour mask (white = detected trim pixels)
  - *_zone.jpg   : the computed board zone polygon overlaid in semi-transparent green
  - *_overlay.jpg: final composite — board zone + trim centreline

Usage:
    python3 debug_trim_detector.py
"""

import cv2
import numpy as np
import os

from src.calibration.rink_calibrator import RinkCalibrator
from src.calibration.trim_detector import TrimDetector

FRAME_DIR  = 'annotation_frames'
OUT_DIR    = 'annotation_frames/debug_output'
os.makedirs(OUT_DIR, exist_ok=True)

rc = RinkCalibrator()
td = TrimDetector()

for fname in sorted(os.listdir(FRAME_DIR)):
    if not fname.endswith('.jpg') or 'annotated' in fname or 'debug' in fname:
        continue

    fpath = os.path.join(FRAME_DIR, fname)
    frame = cv2.imread(fpath)
    if frame is None:
        continue

    stem = os.path.splitext(fname)[0]
    h, w = frame.shape[:2]

    print(f"\n=== {fname} ===")

    # ── Ice detection (used to constrain trim search) ─────────────────────────
    rc.calibrate(frame)
    ice_mask = rc._ice_mask   # may be None if no ice detected

    if ice_mask is not None:
        print(f"  Ice detected: {(ice_mask > 0).sum()} px")
    else:
        print("  ⚠ No ice detected — running trim detection without ice constraint")

    # ── Trim detection ────────────────────────────────────────────────────────
    success = td.detect(frame, ice_mask=ice_mask)
    print(f"  Trim detected: {success}")

    # ── Save trim colour mask ─────────────────────────────────────────────────
    trim_raw = td.get_trim_mask()
    if trim_raw is not None:
        cv2.imwrite(os.path.join(OUT_DIR, f'{stem}_trim.jpg'), trim_raw)

    # ── Build board zone mask ─────────────────────────────────────────────────
    board_mask = td.get_board_mask(frame)
    if board_mask is None and ice_mask is not None:
        print("  Falling back to RinkCalibrator mask")
        board_mask = rc.get_board_mask(frame)

    if board_mask is not None:
        # ── Overlay ───────────────────────────────────────────────────────────
        green = np.zeros_like(frame)
        green[:, :] = (0, 200, 0)
        overlay = frame.copy()
        overlay[board_mask > 0] = (
            frame[board_mask > 0] * 0.45 + green[board_mask > 0] * 0.55
        ).astype(np.uint8)

        # Draw the trim centroid line in yellow
        trim_y = td.get_trim_y()
        if trim_y is not None:
            for x in range(w):
                cy = int(trim_y[x])
                if 0 <= cy < h:
                    cv2.circle(overlay, (x, cy), 1, (0, 220, 220), -1)  # cyan dot per col

        # Draw the board top/bottom lines
        if td.is_detected():
            for x in range(0, w, 4):
                if 0 <= td._board_top_y[x] < h:
                    cv2.circle(overlay, (x, td._board_top_y[x]), 2, (0, 0, 255), -1)  # red = top
                if 0 <= td._board_bot_y[x] < h:
                    cv2.circle(overlay, (x, td._board_bot_y[x]), 2, (0, 255, 255), -1)  # yellow = bot

        # Legend
        cv2.putText(overlay, "GREEN=board zone  RED=top edge  CYAN=trim centroid",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        out_path = os.path.join(OUT_DIR, f'{stem}_overlay.jpg')
        cv2.imwrite(out_path, overlay)
        print(f"  Saved → {out_path}")
    else:
        print("  ⚠ No board mask generated")

print("\nDone! Open annotation_frames/debug_output/ to review results.")
