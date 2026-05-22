"""
debug_board_mask.py

Saves annotated frames showing:
  - Detected red/blue rink lines (colored dots)
  - Projected board mask overlay (semi-transparent green)
  - Player mask overlay (semi-transparent red)

Usage:
  python debug_board_mask.py --source "src/2026-05-12 21-12-18.mp4"
"""
import argparse
import cv2
import numpy as np

from src.inference.model_runner import ModelRunner

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--frames", type=int, default=5,
                    help="Number of debug frames to save")
args = parser.parse_args()

cap = cv2.VideoCapture(args.source)
runner = ModelRunner()

saved = 0
frame_idx = 0

while saved < args.frames:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    board_mask  = runner.get_board_mask(frame)
    player_mask = runner.get_player_mask(frame)

    # ── Build overlay ────────────────────────────────────────────────────────
    overlay = frame.copy()

    # Board mask → semi-transparent green
    board_color = np.zeros_like(frame)
    board_color[board_mask > 0] = (0, 200, 0)
    cv2.addWeighted(board_color, 0.45, overlay, 1.0, 0, overlay)

    # Player mask → semi-transparent red
    player_color = np.zeros_like(frame)
    player_color[player_mask > 0] = (0, 0, 220)
    cv2.addWeighted(player_color, 0.45, overlay, 1.0, 0, overlay)

    # Draw board mask contour
    contours, _ = cv2.findContours(board_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Calibration status label
    label = ("CALIBRATED (geometry)" if runner._calibrated
             else "FALLBACK (HSV)")
    color = (0, 255, 0) if runner._calibrated else (0, 140, 255)
    cv2.putText(overlay, f"Frame {frame_idx}: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    out_path = f"debug_frame_{frame_idx:04d}.jpg"
    cv2.imwrite(out_path, overlay)
    print(f"Saved {out_path}  [{label}]")
    saved += 1

cap.release()
print("Done.")
