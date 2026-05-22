"""
Stitch a full rink panorama from a panning video using robust heuristics.
This integrates the ice-masking and template-matching fallbacks developed 
for the sideline screenshots into a full video pipeline.
"""

import cv2
import numpy as np
import glob
import os
import argparse

# ─── CONFIG ──────────────────────────────────────────────────────────────────
# Default config for overhead/broadcast panning
FRAME_STEP = 30         # Extract 1 frame per second (at 30fps)
SCALE = 0.5             # Downscale for processing speed
CANVAS_W = 10000
CANVAS_H = 3000
OFFSET_X = 5000         # start in the middle so we can go left or right
OFFSET_Y = 500
# ─────────────────────────────────────────────────────────────────────────────

def sharpness(img):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def extract_best_frames(video_path, step, scale):
    """Extract one sharp keyframe per interval of `step` frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    frames = []
    window = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # We process at 30 fps.
        if i % 3 == 0:  # slight optimization: check sharpness on every 3rd frame
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            window.append(small)
            
        if len(window) == step // 3:
            best = max(window, key=sharpness)
            frames.append(best)
            window = []
        i += 1

    if window:
        best = max(window, key=sharpness)
        frames.append(best)

    cap.release()
    print(f"Extracted {len(frames)} keyframes.")
    return frames

def ice_mask(gray):
    """
    Mask targeting the ice surface strip: below the boards, above the crowd/bottom edge.
    For standard broadcast: ice is roughly y=35%..90%
    """
    h, w = gray.shape
    m = np.zeros((h, w), dtype=np.uint8)
    # Upper boards strip
    m[int(h*0.35):int(h*0.45), :] = 255  
    # Ice surface (unique markings)
    m[int(h*0.45):int(h*0.90), :] = 255
    return m

def detect_and_filter_kp(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=5000, contrastThreshold=0.03)
    mask = ice_mask(gray)
    kp, des = sift.detectAndCompute(gray, mask)
    return kp, des, gray

def match_pair(img_src, img_dst, label=""):
    kp1, des1, g1 = detect_and_filter_kp(img_dst)
    kp2, des2, g2 = detect_and_filter_kp(img_src)
    
    if des1 is None or des2 is None: return None
    
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches12 = bf.knnMatch(des1, des2, k=2)
    good12 = [m for m, n in matches12 if m.distance < 0.72 * n.distance]
    
    matches21 = bf.knnMatch(des2, des1, k=2)
    good21 = {n.trainIdx: n for m, n in matches21 if m.distance < 0.72 * n.distance}
    
    verified = [m for m in good12 if m.queryIdx in {g.queryIdx for g in good21.values()}]
    
    if len(verified) < 6: verified = good12
    if len(verified) < 6: return None
    
    src_pts = np.float32([kp2[m.trainIdx].pt for m in verified]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in verified]).reshape(-1, 1, 2)
    
    M, inl = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                          ransacReprojThreshold=5.0)
    
    if M is None: return None
    
    ni = int(inl.sum()) if inl is not None else 0
    print(f"  {label}: {ni} inliers  tx={M[0,2]:.1f}  ty={M[1,2]:.1f}")
    
    if abs(M[0,2]) < 5.0 and abs(M[1,2]) < 5.0:
        print(f"  {label}: WARNING - near-zero transform, likely matched repeated ads!")
        return None
        
    return np.vstack([M, [0, 0, 1]]).astype(np.float64)

def stitch_pair_manual(img_left, img_right):
    """
    Template matching fallback.
    Assumes img_right is a panning continuation to the right of img_left.
    Takes a strip from the right of img_left, searches in the left of img_right.
    """
    h, w = img_left.shape[:2]
    strip_w = int(w * 0.4)
    
    # Template: right side of left image (ice + lower boards)
    template = img_left[int(h*0.35):int(h*0.85), w-strip_w:]
    # Search: left side of right image
    search = img_right[int(h*0.35):int(h*0.85), :int(w*0.8)]
    
    tg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    
    if tg.shape[0] > sg.shape[0] or tg.shape[1] > sg.shape[1]:
        return None, None

    result = cv2.matchTemplate(sg, tg, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    print(f"  Template match: score={max_val:.3f}")
    if max_val < 0.25: return None, None
    
    overlap_in_right = max_loc[0]
    left_start_in_right = overlap_in_right - (w - strip_w)
    tx = -left_start_in_right
    ty = max_loc[1]
    
    return tx, ty

def blend(canvas, wc, img, H, cw, ch):
    warped = cv2.warpPerspective(img, H, (cw, ch))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    w = dist[..., np.newaxis].astype(np.float32)
    canvas += warped.astype(np.float32) * w
    wc += w

def crop_canvas(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)
    c = cv2.findNonZero(th)
    if c is None: return img
    x, y, w, h = cv2.boundingRect(c)
    return img[max(0,y):y+h, max(0,x):x+w]

def main(video_path, out_path):
    print(f"=== Processing video: {video_path} ===")
    frames = extract_best_frames(video_path, FRAME_STEP, SCALE)
    if len(frames) < 2:
        print("Not enough frames.")
        return

    H_list = [None] * len(frames)
    H_list[0] = np.array([[1,0,OFFSET_X],[0,1,OFFSET_Y],[0,0,1]], dtype=np.float64)
    h, w = frames[0].shape[:2]

    for i in range(1, len(frames)):
        print(f"\nFrame {i}/{len(frames)-1}:")
        img_cur = frames[i]
        img_prev = frames[i-1]
        
        # 1. Try SIFT on ice
        M = match_pair(img_cur, img_prev, "SIFT")
        
        # 2. Try Template Matching fallback
        if M is None:
            print("  Falling back to template matching...")
            # We don't know the pan direction, try left-to-right pan
            tx, ty = stitch_pair_manual(img_prev, img_cur)
            if tx is not None:
                # If tx > 0, it means img_cur is to the right of img_prev
                prev_tx = H_list[i-1][0, 2]
                prev_ty = H_list[i-1][1, 2]
                M = np.array([[1,0,prev_tx + w - tx],[0,1,prev_ty + ty],[0,0,1]], dtype=np.float64)
                
        # 3. Geometric fallback (if everything fails, assume pan continues)
        if M is None and i > 1 and H_list[i-1] is not None and H_list[i-2] is not None:
            print("  Geometric fallback...")
            # Use velocity from previous pair
            vx = H_list[i-1][0,2] - H_list[i-2][0,2]
            vy = H_list[i-1][1,2] - H_list[i-2][1,2]
            if abs(vx) > 0:
                M = np.array([[1,0,H_list[i-1][0,2]+vx],[0,1,H_list[i-1][1,2]+vy],[0,0,1]], dtype=np.float64)
                
        if M is not None:
            if M.shape == (3,3) and M[2,2] == 1:
                 # It's an absolute H
                 H_list[i] = M
            else:
                 # It's a relative M
                 H_list[i] = H_list[i-1] @ M
        else:
            print(f"  Failed to stitch frame {i}")

    print("\n=== Building Canvas ===")
    cw, ch = CANVAS_W, CANVAS_H
    canvas = np.zeros((ch, cw, 3), dtype=np.float32)
    wc = np.zeros((ch, cw, 1), dtype=np.float32)

    stitched_count = 0
    for i, img in enumerate(frames):
        if H_list[i] is not None:
            blend(canvas, wc, img, H_list[i], cw, ch)
            stitched_count += 1

    wc[wc == 0] = 1.0
    result = (canvas / wc).astype(np.uint8)
    result = crop_canvas(result)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nSaved {stitched_count} frames to → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="src/2026-05-12 21-12-59.mp4")
    parser.add_argument("--out", default="/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/full_rink_panorama.jpg")
    args = parser.parse_args()
    main(args.video, args.out)
