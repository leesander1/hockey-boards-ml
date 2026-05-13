"""
Stitch all 6 screenshots into a full rink panorama.

After inspecting the images, the spatial order left-to-right is:
  ss4 (10.36.20) = LEFT END (goalie in net, end boards)
  ss3 (10.36.10) = LEFT ZONE (faceoff circle, Toyota/Score Cash Back boards)  
  ss5 (10.36.27) = LEFT-CENTER (center ice logo, APP boards)
  ss2 (10.08.49) = CENTER-RIGHT (center ice, blue line right side)
  ss1 (10.08.46) = RIGHT-CENTER (right faceoff dot, Energizer/Caesars boards)
  ss0 (10.08.20) = RIGHT ZONE (far right end, possibly net)

Strategy:
  - Chain: each image is registered to its spatial neighbor
  - Use SIFT on boards strip only
  - Affine transform (no perspective distortion per frame)
  - Distance-weighted blend for smooth seams
"""

import cv2
import numpy as np
import glob, os

OUT = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_3_photos_final_homography.jpg"

BOARD_TOP = 0.03
BOARD_BOT = 0.45
SCALE = 0.4   # 40% of original (3404 → ~1360px wide)


def load_all():
    files = sorted(glob.glob("src/Screenshot*.png"))
    imgs = []
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (int(img.shape[1]*SCALE), int(img.shape[0]*SCALE)))
        imgs.append((f, img))
        print(f"  [{len(imgs)-1}] {f.split('at ')[1].strip()}: {img.shape}")
    return imgs


def board_mask(gray):
    h, w = gray.shape
    m = np.zeros((h, w), dtype=np.uint8)
    m[int(h*BOARD_TOP):int(h*BOARD_BOT), :] = 255
    return m


def match_affine(img_src, img_dst, label=""):
    """Returns 3x3 H such that img_src pixels → img_dst coordinates."""
    sift = cv2.SIFT_create(nfeatures=5000)
    g_dst = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)
    g_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    kp1, des1 = sift.detectAndCompute(g_dst, board_mask(g_dst))
    kp2, des2 = sift.detectAndCompute(g_src, board_mask(g_src))
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        print(f"  {label}: insufficient keypoints")
        return None
    flann = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 100})
    raw = flann.knnMatch(des1, des2, k=2)
    good = [m for m, n in raw if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        print(f"  {label}: not enough good matches ({len(good)})")
        return None
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                              ransacReprojThreshold=4.0)
    if M is None:
        print(f"  {label}: estimateAffinePartial2D failed")
        return None
    n_in = int(inliers.sum()) if inliers is not None else 0
    print(f"  {label}: {n_in} inliers  tx={M[0,2]:.1f}  ty={M[1,2]:.1f}")
    return np.vstack([M, [0, 0, 1]]).astype(np.float64)


def blend(canvas, wc, img, H, cw, ch):
    warped = cv2.warpPerspective(img, H, (cw, ch))
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
    w = dist[..., np.newaxis].astype(np.float32)
    canvas += warped.astype(np.float32) * w
    wc += w


def crop(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(g, 1, 255, cv2.THRESH_BINARY)
    c = cv2.findNonZero(th)
    if c is None: return img
    x, y, w, h = cv2.boundingRect(c)
    return img[max(0,y-15):y+h+15, max(0,x-15):x+w+15]


def main():
    print("Loading images...")
    imgs = load_all()

    # imgs order by filename sort = [0]=10.08.20, [1]=10.08.46, [2]=10.08.49, [3]=10.36.10, [4]=10.36.20, [5]=10.36.27
    # Spatial left→right:
    spatial = [
        imgs[4],   # 10.36.20 = LEFT END (ANA net)
        imgs[3],   # 10.36.10 = LEFT ZONE
        imgs[5],   # 10.36.27 = LEFT-CENTER
        imgs[2],   # 10.08.49 = CENTER
        imgs[1],   # 10.08.46 = RIGHT-CENTER
        imgs[0],   # 10.08.20 = RIGHT ZONE
    ]

    h, w = spatial[0][1].shape[:2]
    cw = w * 7
    ch = int(h * 2.0)
    # Place leftmost image at left with some padding
    offset_x = 50
    offset_y = int(h * 0.3)

    H_anchors = [None] * len(spatial)
    H_anchors[0] = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float64)

    print("\nChaining transforms left→right...")
    for i in range(1, len(spatial)):
        label, img_cur = spatial[i]
        _, img_prev = spatial[i-1]
        lname = f"{label.split('at ')[1].strip()} → {spatial[i-1][0].split('at ')[1].strip()}"
        M = match_affine(img_cur, img_prev, lname)
        if M is not None:
            H_anchors[i] = H_anchors[i-1] @ M
        else:
            print(f"  WARNING: Could not chain image {i}, trying direct match to anchor 0")
            M2 = match_affine(img_cur, spatial[0][1], f"img{i}→anchor0")
            if M2 is not None:
                H_anchors[i] = H_anchors[0] @ M2
            else:
                print(f"  SKIPPING image {i}")

    print("\nBlending onto canvas...")
    canvas = np.zeros((ch, cw, 3), dtype=np.float32)
    wc = np.zeros((ch, cw, 1), dtype=np.float32)

    for i, (fname, img) in enumerate(spatial):
        if H_anchors[i] is not None:
            blend(canvas, wc, img, H_anchors[i], cw, ch)
            print(f"  Blended [{i}] {fname.split('at ')[1].strip()}")
        else:
            print(f"  Skipped [{i}] {fname.split('at ')[1].strip()}")

    wc[wc == 0] = 1.0
    result = (canvas / wc).astype(np.uint8)
    result = crop(result)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, result, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"\nSaved → {OUT}  ({result.shape[1]}x{result.shape[0]})")


if __name__ == "__main__":
    main()
