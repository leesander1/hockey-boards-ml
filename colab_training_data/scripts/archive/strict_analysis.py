import cv2
import numpy as np
import glob
import os
from collections import defaultdict

def analyze_strict(path):
    img = cv2.imread(path)
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    total_pixels = h * w

    # Red strict: H in [0,8] or [172,180], S>=220, V>=180
    mask_red1 = cv2.inRange(hsv, np.array([0, 220, 180]), np.array([8, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([172, 220, 180]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Yellow strict: H in [20,33], S>=220, V>=180
    mask_yellow = cv2.inRange(hsv, np.array([20, 220, 180]), np.array([33, 255, 255]))

    red_frac = np.sum(mask_red > 0) / total_pixels
    yellow_frac = np.sum(mask_yellow > 0) / total_pixels

    # Valid col frac: columns where both masks exist
    col_has_red = np.any(mask_red > 0, axis=0)
    col_has_yellow = np.any(mask_yellow > 0, axis=0)
    valid_cols = np.sum(col_has_red & col_has_yellow)
    valid_col_frac = valid_cols / w

    return red_frac, yellow_frac, valid_col_frac

files = sorted(glob.glob("annotation_frames/*.jpg") + glob.glob("annotation_frames/*.png"))
data_by_ext = defaultdict(list)

print(f"{'File':<30} | {'R-Strict%':<10} | {'Y-Strict%':<10} | {'VCol%'}")
print("-" * 65)

for f in files:
    res = analyze_strict(f)
    if res:
        ext = os.path.splitext(f)[1].lower()
        data_by_ext[ext].append(res)
        name = os.path.basename(f)
        print(f"{name:<30} | {res[0]:.5f} | {res[1]:.5f} | {res[2]:.5f}")

print("\nSummary (Median and Range):")
print(f"{'Ext':<5} | {'R-Strict% (Med)':<15} | {'Y-Strict% (Med)':<15} | {'VCol% (Med)':<15}")
print("-" * 65)
for ext, values in data_by_ext.items():
    arr = np.array(values)
    medians = np.median(arr, axis=0)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    print(f"{ext:<5} | {medians[0]:.5f}         | {medians[1]:.5f}         | {medians[2]:.5f}")
    print(f"{'range':<5} | {mins[0]:.5f}-{maxs[0]:.5f} | {mins[1]:.5f}-{maxs[1]:.5f} | {mins[2]:.5f}-{maxs[2]:.5f}")
