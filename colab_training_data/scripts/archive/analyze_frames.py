import cv2
import numpy as np
import glob
import os
from collections import defaultdict

# Red/Yellow Thresholds (from typical detector)
LOWER_RED1 = np.array([0, 100, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 100, 50])
UPPER_RED2 = np.array([180, 255, 255])
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([30, 255, 255])

def analyze(path):
    img = cv2.imread(path)
    if img is None: return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]
    total_pixels = h * w

    mask_red = cv2.bitwise_or(cv2.inRange(hsv, LOWER_RED1, UPPER_RED1), cv2.inRange(hsv, LOWER_RED2, UPPER_RED2))
    mask_yellow = cv2.inRange(hsv, LOWER_YELLOW, UPPER_YELLOW)

    red_frac = np.sum(mask_red > 0) / total_pixels
    yellow_frac = np.sum(mask_yellow > 0) / total_pixels

    valid_cols = np.sum((np.any(mask_red > 0, axis=0)) & (np.any(mask_yellow > 0, axis=0)))
    valid_col_frac = valid_cols / w

    def get_max_cc(mask):
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num <= 1: return 0, 0, 0
        max_idx = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        return stats[max_idx, cv2.CC_STAT_WIDTH], stats[max_idx, cv2.CC_STAT_HEIGHT], stats[max_idx, cv2.CC_STAT_AREA]

    rw, rh, ra = get_max_cc(mask_red)
    yw, yh, ya = get_max_cc(mask_yellow)

    return red_frac, yellow_frac, valid_col_frac, ra, ya

files = glob.glob("annotation_frames/*.jpg") + glob.glob("annotation_frames/*.png")
results = defaultdict(list)

for f in files:
    ext = os.path.splitext(f)[1].lower()
    res = analyze(f)
    if res: results[ext].append(res)

print(f"{'Ext':<5} | {'Red%':<6} | {'Yel%':<6} | {'VCol%':<6} | {'R-Area':<7} | {'Y-Area':<7}")
print("-" * 45)
for ext, data in results.items():
    arr = np.array(data)
    means = np.mean(arr, axis=0)
    mins = np.min(arr, axis=0)
    maxs = np.max(arr, axis=0)
    print(f"{ext:<5} | {means[0]:.3f} | {means[1]:.3f} | {means[2]:.3f} | {int(means[3]):<7} | {int(means[4]):<7}")
    print(f"{'range':<5} | {mins[0]:.3f}-{maxs[0]:.3f} | {mins[1]:.3f}-{maxs[1]:.3f} | {mins[2]:.3f}-{maxs[2]:.3f}")
