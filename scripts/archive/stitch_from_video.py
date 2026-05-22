"""
Stitch a rink panorama from a panning video.

Key insight: The blurring in the previous attempt came from averaging
too many frames (including mid-action frames with moving players).

New approach:
  1. Use sparse keyframes (1 per ~2 sec) — enough to track camera motion
  2. Find the affine transform between ADJACENT keyframes (boards only)
  3. Compound transforms so every frame maps to a common canvas
  4. Use "last writer wins" compositing (not alpha blend) so we get sharp frames
  5. Optionally use the frame with LEAST motion blur (sharpness score) per interval
"""

import cv2
import numpy as np
import os

# ─── CONFIG ──────────────────────────────────────────────────────────────────
VIDEO_PATH = "src/2026-05-12 21-12-18.mp4"
OUTPUT_PATH = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_rink_extended.jpg"

# Sample one frame per this many frames (at 30fps, 60 = 2 sec)
FRAME_STEP = 60        
SCALE = 0.75            # keep higher res for sharper result

CANVAS_W = 8000
CANVAS_H = 1600
OFFSET_X = 3500         # start in the middle so we can go left or right
OFFSET_Y = 300

# Board strip: only use features from the boards/glass (top portion of frame)
BOARD_TOP_FRAC = 0.02   # skip very top (scoreboard)
BOARD_BOT_FRAC = 0.38   # stop before ice
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
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        window.append(small)
        if len(window) == step:
            # pick sharpest frame in window
            best = max(window, key=sharpness)
            frames.append(best)
            window = []
        i += 1

    if window:  # leftover
        best = max(window, key=sharpness)
        frames.append(best)

    cap.release()
    print(f"Extracted {len(frames)} keyframes.")
    return frames


def get_board_mask(gray):
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    y0 = int(h * BOARD_TOP_FRAC)
    y1 = int(h * BOARD_BOT_FRAC)
    mask[y0:y1, :] = 255
    return mask


def find_affine(img1, img2):
    """Affine (tx, ty, rotation, scale) from img2 → img1 coordinates."""
    sift = cv2.SIFT_create(nfeatures=3000)

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.detectAndCompute(g1, get_board_mask(g1))
    kp2, des2 = sift.detectAndCompute(g2, get_board_mask(g2))

    if des1 is None or des2 is None or len(kp1) < 6 or len(kp2) < 6:
        return None

    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    raw = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]

    if len(good) < 6:
        return None

    src = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=3.0)
    if M is None:
        return None

    # Reject clearly bad transforms
    tx, ty = abs(M[0, 2]), abs(M[1, 2])
    if tx > img1.shape[1] * 0.7 or ty > img1.shape[0] * 0.5:
        print(f"    Rejected (translation too large: tx={M[0,2]:.1f} ty={M[1,2]:.1f})")
        return None

    inlier_count = int(inliers.sum()) if inliers is not None else 0
    print(f"    Matched: {inlier_count} inliers  tx={M[0,2]:.1f}  ty={M[1,2]:.1f}")

    return np.vstack([M, [0, 0, 1]]).astype(np.float64)


def crop_to_content(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    pad = 20
    return img[max(0,y-pad):y+h+pad, max(0,x-pad):x+w+pad]


def main():
    frames = extract_best_frames(VIDEO_PATH, FRAME_STEP, SCALE)

    # Start canvas
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)

    # Cumulative transform (frame[i] → canvas coordinates)
    H_cum = np.array([[1,0,OFFSET_X],[0,1,OFFSET_Y],[0,0,1]], dtype=np.float64)

    # Paint frame 0
    warped = cv2.warpPerspective(frames[0], H_cum, (CANVAS_W, CANVAS_H))
    mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
    canvas[mask] = warped[mask]

    last_good = frames[0]
    last_good_H = H_cum.copy()
    stitched = 1

    for i in range(1, len(frames)):
        print(f"Frame {i}/{len(frames)-1}:")
        M = find_affine(last_good, frames[i])
        if M is None:
            print(f"  → skip (no match)")
            continue

        H_cum = last_good_H @ M
        warped = cv2.warpPerspective(frames[i], H_cum, (CANVAS_W, CANVAS_H))
        mask = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0
        # "Last writer wins" — sharp frames only, no blurry average
        canvas[mask] = warped[mask]

        last_good = frames[i]
        last_good_H = H_cum.copy()
        stitched += 1

    print(f"\nTotal stitched: {stitched}/{len(frames)}")
    result = crop_to_content(canvas)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    cv2.imwrite(OUTPUT_PATH, result)
    print(f"Saved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
