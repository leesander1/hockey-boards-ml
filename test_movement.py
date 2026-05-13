import cv2
import numpy as np
import glob

def get_movement(v):
    cap = cv2.VideoCapture(v)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    # Extract 1 frame per second
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if i % int(fps) == 0:
            gray = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2GRAY)
            frames.append(gray)
        i += 1
    cap.release()

    if len(frames) < 2: return 0
    
    # Calculate average movement between 1-sec apart frames
    sift = cv2.SIFT_create()
    total_tx = 0
    for i in range(1, len(frames)):
        kp1, des1 = sift.detectAndCompute(frames[i-1], None)
        kp2, des2 = sift.detectAndCompute(frames[i], None)
        if des1 is None or des2 is None: continue
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) >= 4:
            src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            M, _ = cv2.estimateAffinePartial2D(dst, src) # dst -> src
            if M is not None:
                total_tx += abs(M[0, 2])
    print(f"{v}: {len(frames)} sec, total horizontal movement: {total_tx:.1f} px")

videos = sorted(glob.glob("src/*.mp4"))
for v in videos:
    get_movement(v)
