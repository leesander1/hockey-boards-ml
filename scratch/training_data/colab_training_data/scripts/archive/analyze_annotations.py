"""
Reads annotated frames (PNG with red/yellow lines drawn by human) and extracts
calibrated board geometry. Handles cases where lines may be swapped (always takes
top line as board-top regardless of color).

Reports:
  - Fitted linear model: board_height_px = TOP + (y/h) * (BOTTOM - TOP)
  - Trim HSV statistics from pixels between the two annotation lines
  - Saves annotation_frames/calibration.json
"""

import cv2
import numpy as np
import os
import json

ANNOTATION_DIR = 'annotation_frames'

FRAMES = [
    ('v1_f010_overhead_end_zone.png', 'v1_f010_overhead_end_zone.jpg'),
    ('v1_f317_overhead_end_zone.png', 'v1_f317_overhead_end_zone.jpg'),
    ('v1_f634_overhead_end_zone.png', 'v1_f634_overhead_end_zone.jpg'),
    ('v3_f010_overhead_full.png',     'v3_f010_overhead_full.jpg'),
    ('v3_f200_overhead_full.png',     'v3_f200_overhead_full.jpg'),
    ('v4_f010_overhead_near.png',     'v4_f010_overhead_near.jpg'),
]

# Annotation line colors in HSV
RED_LO1 = np.array([0,   150, 80]);  RED_HI1 = np.array([8,   255, 255])
RED_LO2 = np.array([170, 150, 80]);  RED_HI2 = np.array([180, 255, 255])
YEL_LO  = np.array([18,  120, 100]); YEL_HI  = np.array([40,  255, 255])


def col_centroid(mask, h):
    rows  = np.arange(h, dtype=np.float32)
    count = mask.sum(axis=0).astype(np.float32)
    wsum  = (mask.astype(np.float32) * rows[:, None]).sum(axis=0)
    valid = count >= 3
    cy    = np.where(valid, wsum / np.where(count > 0, count, 1), np.nan)
    return cy, valid


all_measurements  = []   # (y_frac, board_h_px)
board_px_samples  = []   # raw pixel colours inside the board region per frame

for ann_name, orig_name in FRAMES:
    ann  = cv2.imread(os.path.join(ANNOTATION_DIR, ann_name))
    orig = cv2.imread(os.path.join(ANNOTATION_DIR, orig_name))
    if ann is None:
        print(f"⚠  Missing {ann_name}"); continue

    h, w = ann.shape[:2]
    ann_hsv = cv2.cvtColor(ann, cv2.COLOR_BGR2HSV)

    red_mask = cv2.bitwise_or(
        cv2.inRange(ann_hsv, RED_LO1, RED_HI1),
        cv2.inRange(ann_hsv, RED_LO2, RED_HI2))
    yel_mask = cv2.inRange(ann_hsv, YEL_LO, YEL_HI)

    red_y, red_v = col_centroid(red_mask, h)
    yel_y, yel_v = col_centroid(yel_mask, h)

    print(f"\n=== {ann_name} ===")
    print(f"  Red  coverage {red_v.mean():.0%}  mean-y={np.nanmean(red_y):.0f}")
    print(f"  Yel  coverage {yel_v.mean():.0%}  mean-y={np.nanmean(yel_y):.0f}")

    # Always: top_y = whichever line is higher (smaller y), bot_y = lower
    both = red_v & yel_v
    if both.sum() < 20:
        print("  ⚠ Not enough overlap — skipping"); continue

    top_y = np.where(red_y[both] < yel_y[both], red_y[both], yel_y[both])
    bot_y = np.where(red_y[both] > yel_y[both], red_y[both], yel_y[both])
    bh    = bot_y - top_y          # must be positive
    ref_y = bot_y                  # board bottom y-position

    # Filter to physically plausible range  (5 – 250 px)
    ok = (bh > 5) & (bh < 250)
    bh, ref_y = bh[ok], ref_y[ok]

    if len(bh) == 0:
        print("  ⚠ No valid board-height measurements"); continue

    print(f"  Board height: {bh.min():.0f}–{bh.max():.0f} px  (mean {bh.mean():.0f})")
    for b, y in zip(bh[::4], ref_y[::4]):
        all_measurements.append((float(y / h), float(b)))

    # Sample original pixels inside the board strip
    if orig is not None:
        orig_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
        top_arr = col_centroid(red_mask, h)[0] if np.nanmean(red_y) < np.nanmean(yel_y) else col_centroid(yel_mask, h)[0]
        bot_arr = col_centroid(yel_mask, h)[0] if np.nanmean(red_y) < np.nanmean(yel_y) else col_centroid(red_mask, h)[0]
        for x in range(0, w, 8):
            ty = int(np.nan_to_num(top_arr[x], nan=0))
            by = int(np.nan_to_num(bot_arr[x], nan=0))
            if ty >= by or by - ty > 200: continue
            for ry in range(max(0, ty), min(h, by)):
                board_px_samples.append(orig_hsv[ry, x].tolist())

# ── Linear fit ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
if len(all_measurements) > 50:
    meas = np.array(all_measurements)
    A    = np.column_stack([np.ones(len(meas)), meas[:, 0]])
    coef, *_ = np.linalg.lstsq(A, meas[:, 1], rcond=None)
    a, b      = coef
    top_px    = int(max(5,  round(a)))
    bot_px    = int(max(10, round(a + b)))

    resid = meas[:, 1] - (A @ coef)
    r2    = 1 - np.var(resid) / np.var(meas[:, 1])
    print(f"LINEAR FIT  R²={r2:.3f}")
    print(f"  BOARD_HEIGHT_TOP_PX    = {top_px}   (y ~ 0, far boards)")
    print(f"  BOARD_HEIGHT_BOTTOM_PX = {bot_px}   (y ~ h, near boards)")
else:
    print("Not enough data"); top_px, bot_px = 22, 90

# ── Board pixel colour stats ──────────────────────────────────────────────────
print("\nBOARD PIXEL HSV STATISTICS (original frames, between annotation lines):")
calibration = {
    "board_height_top_px":    top_px,
    "board_height_bottom_px": bot_px,
}
if board_px_samples:
    arr = np.array(board_px_samples, dtype=np.float32)
    for ch, name in enumerate(['H', 'S', 'V']):
        p5, p50, p95 = np.percentile(arr[:, ch], [5, 50, 95])
        print(f"  {name}: p5={p5:.0f}  median={p50:.0f}  p95={p95:.0f}")

    # The board face itself is generally low-S, high-V (white/light-coloured)
    calibration["board_face_s_max"] = int(np.percentile(arr[:, 1], 80))
    calibration["board_face_v_min"] = int(np.percentile(arr[:, 2], 10))

# ── Trim (kickplate) stats ─────────────────────────────────────────────────────
# Sample a 3-px band just above the bottom annotation line in the originals
trim_samples = []
for ann_name, orig_name in FRAMES:
    ann  = cv2.imread(os.path.join(ANNOTATION_DIR, ann_name))
    orig = cv2.imread(os.path.join(ANNOTATION_DIR, orig_name))
    if ann is None or orig is None: continue
    h, w = ann.shape[:2]
    ann_hsv  = cv2.cvtColor(ann,  cv2.COLOR_BGR2HSV)
    orig_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
    red_y, red_v = col_centroid(cv2.bitwise_or(
        cv2.inRange(ann_hsv, RED_LO1, RED_HI1),
        cv2.inRange(ann_hsv, RED_LO2, RED_HI2)), h)
    yel_y, yel_v = col_centroid(cv2.inRange(ann_hsv, YEL_LO, YEL_HI), h)
    # bottom line = whichever is lower
    for x in range(0, w, 6):
        if not (red_v[x] and yel_v[x]): continue
        bot = int(max(red_y[x], yel_y[x]))
        for dy in range(-5, 5):
            ry = bot + dy
            if 0 <= ry < h:
                trim_samples.append(orig_hsv[ry, x].tolist())

print("\nTRIM REGION HSV (5px around bottom annotation line):")
if trim_samples:
    tarr = np.array(trim_samples, dtype=np.float32)
    for ch, name in enumerate(['H', 'S', 'V']):
        p5, p50, p95 = np.percentile(tarr[:, ch], [5, 50, 95])
        print(f"  {name}: p5={p5:.0f}  median={p50:.0f}  p95={p95:.0f}")
    calibration["trim_h_lo"] = max(0,   int(np.percentile(tarr[:,0],  5)) - 3)
    calibration["trim_h_hi"] = min(180, int(np.percentile(tarr[:,0], 95)) + 3)
    calibration["trim_s_lo"] = max(0,   int(np.percentile(tarr[:,1], 15)) - 10)
    calibration["trim_v_lo"] = max(0,   int(np.percentile(tarr[:,2], 15)) - 15)

out = 'annotation_frames/calibration.json'
with open(out, 'w') as f:
    json.dump(calibration, f, indent=2)
print(f"\n→ Saved: {out}")
print(json.dumps(calibration, indent=2))
