import cv2
import glob
import numpy as np

for v in sorted(glob.glob("src/*.mp4")):
    cap = cv2.VideoCapture(v)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames // 2)
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate how much of the middle is white (ice)
        mid_strip = gray[:, w//2-50:w//2+50]
        white_pixels = np.sum(mid_strip > 200, axis=1)
        ice_rows = np.where(white_pixels > 80)[0]
        if len(ice_rows) > 0:
            ice_start = ice_rows[0] / h
            ice_end = ice_rows[-1] / h
            print(f"{v}: Ice from {ice_start:.2f} to {ice_end:.2f} ({(ice_end-ice_start)*100:.1f}%)")
    cap.release()
