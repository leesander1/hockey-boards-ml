import cv2
import numpy as np
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

orig = cv2.imread('annotation_frames/v1_f010_overhead_end_zone.jpg')
ann = cv2.imread('annotation_frames/v1_f010_overhead_end_zone.png')

RED_LO1 = np.array([0,   150, 150], np.uint8);  RED_HI1 = np.array([10,  255, 255], np.uint8)
RED_LO2 = np.array([170, 150, 150], np.uint8);  RED_HI2 = np.array([180, 255, 255], np.uint8)
YEL_LO  = np.array([18,  150, 150], np.uint8);  YEL_HI  = np.array([35,  255, 255], np.uint8)

h, w = ann.shape[:2]
hsv = cv2.cvtColor(ann, cv2.COLOR_BGR2HSV)

red = cv2.bitwise_or(
    cv2.inRange(hsv, RED_LO1, RED_HI1),
    cv2.inRange(hsv, RED_LO2, RED_HI2))
yel = cv2.inRange(hsv, YEL_LO, YEL_HI)

cv2.imwrite('debug_red.png', red)
cv2.imwrite('debug_yel.png', yel)
cv2.imwrite('debug_ann.png', ann)

top_y = np.full(w, np.nan, dtype=np.float32)
bot_y = np.full(w, np.nan, dtype=np.float32)

for col in range(w):
    r_ys = np.where(red[:, col] > 0)[0]
    y_ys = np.where(yel[:, col] > 0)[0]
    if r_ys.size: top_y[col] = float(r_ys.min())
    if y_ys.size: bot_y[col] = float(y_ys.max())

# plot the top_y and bot_y
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.plot(top_y, color='red', label='top_y')
plt.plot(bot_y, color='yellow', label='bot_y')
plt.legend()
plt.savefig('debug_lines_raw.png')
plt.close()

valid = ~(np.isnan(top_y) | np.isnan(bot_y))
if valid.sum() >= w * 0.1:
    lo, hi = np.where(valid)[0][[0, -1]]
    top_y[:lo] = np.nan; top_y[hi+1:] = np.nan
    bot_y[:lo] = np.nan; bot_y[hi+1:] = np.nan

    def interp_nans(a):
        mask = np.isnan(a)
        if mask.all(): return a
        xs = np.arange(len(a))
        a[mask] = np.interp(xs[mask], xs[~mask], a[~mask])
        return a

    sw = max(11, w // 12) | 1
    top_y_smooth = median_filter(interp_nans(top_y), size=sw).astype(np.int32)
    bot_y_smooth = median_filter(interp_nans(bot_y), size=sw).astype(np.int32)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.plot(top_y_smooth, color='red', label='top_y smoothed')
    plt.plot(bot_y_smooth, color='yellow', label='bot_y smoothed')
    plt.legend()
    plt.savefig('debug_lines_smooth.png')
    plt.close()
