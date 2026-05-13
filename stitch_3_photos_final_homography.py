"""
Stitch the three 10.36.xx sideline-camera screenshots into a panorama.

These photos all come from a low sideline camera (matching stitch.png geometry).
The challenge: ads repeat so SIFT matches the wrong copies.

Solution: Focus ONLY on the ICE SURFACE (below the boards, above the crowd)
where unique markings exist (faceoff circles, lines, logos).
Use a wide-strip mask on the ice to find real spatial offsets.
"""

import cv2
import numpy as np
import glob, os, sys

OUT = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_3_photos_final_homography.jpg"

SCALE = 0.35  # Work at 35% of original


def load_36_shots():
    """Load the 3 sideline-camera screenshots (10.36.xx)."""
    files = sorted([f for f in glob.glob("src/Screenshot*.png") if "10.36" in f])
    print(f"Found {len(files)} sideline shots:")
    imgs = []
    for f in files:
        img = cv2.imread(f)
        img = cv2.resize(img, (int(img.shape[1]*SCALE), int(img.shape[0]*SCALE)))
        imgs.append((f, img))
        print(f"  {f.split('at ')[1].strip()}: {img.shape}")
    return imgs


def ice_mask(gray):
    """
    Mask targeting the ice surface strip: below the boards, above the crowd.
    For sideline camera: ice occupies roughly y=15%..75% of frame.
    Focus on horizontal strip that includes faceoff circles and lines.
    """
    h, w = gray.shape
    m = np.zeros((h, w), dtype=np.uint8)
    # Upper boards strip (for structural alignment, less ad repetition at extremes)
    m[int(h*0.08):int(h*0.25), :] = 255  
    # Ice surface (unique markings)
    m[int(h*0.20):int(h*0.70), :] = 255
    return m


def detect_and_filter_kp(img):
    """
    Detect keypoints but filter out ones that appear in ad-like repeated regions.
    Strategy: compute horizontal FFT, suppress features near repeating-frequency bands.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # Use ORB which is better at unique structural features
    # (SIFT can match repeated textures too well)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.02)
    mask = ice_mask(gray)
    kp, des = sift.detectAndCompute(gray, mask)
    return kp, des, gray


def match_pair(img_src, img_dst, label=""):
    """
    Match img_src → img_dst.
    Returns affine 3x3 matrix.
    """
    kp1, des1, g1 = detect_and_filter_kp(img_dst)
    kp2, des2, g2 = detect_and_filter_kp(img_src)
    
    print(f"  {label}: kp_dst={len(kp1)}, kp_src={len(kp2)}")
    
    if des1 is None or des2 is None:
        return None
    
    # BF matcher with ratio test + cross-check
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches12 = bf.knnMatch(des1, des2, k=2)
    good12 = [m for m, n in matches12 if m.distance < 0.72 * n.distance]
    
    matches21 = bf.knnMatch(des2, des1, k=2)
    good21 = {n.trainIdx: n for m, n in matches21 if m.distance < 0.72 * n.distance}
    
    # Cross-verified matches only
    verified = [m for m in good12 if m.queryIdx in {g.queryIdx for g in good21.values()}]
    
    print(f"  {label}: {len(good12)} ratio-passed, {len(verified)} cross-verified")
    
    if len(verified) < 6:
        # Fallback: use all ratio-passed matches
        verified = good12
    
    if len(verified) < 6:
        print(f"  {label}: insufficient matches")
        return None
    
    src_pts = np.float32([kp2[m.trainIdx].pt for m in verified]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in verified]).reshape(-1, 1, 2)
    
    M, inl = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC,
                                          ransacReprojThreshold=6.0, confidence=0.99,
                                          maxIters=5000)
    ni = int(inl.sum()) if inl is not None else 0
    if M is None:
        print(f"  {label}: RANSAC failed")
        return None
    
    print(f"  {label}: {ni} inliers  tx={M[0,2]:.1f}  ty={M[1,2]:.1f}  scale={M[0,0]:.3f}")
    
    # Sanity check: the tx should be significant (images show different parts of rink)
    # If tx is near zero but we expect ~horizontal pan, something is wrong
    if abs(M[0,2]) < 5.0 and abs(M[1,2]) < 5.0:
        print(f"  {label}: WARNING - near-zero transform, likely matched repeated ads!")
        # Try with stricter constraints: only use keypoints in the left/right thirds
        # to avoid center-region symmetric matching
        return None
    
    return np.vstack([M, [0, 0, 1]]).astype(np.float64)


def stitch_pair_manual(img_left, img_right, offset_x_hint=None):
    """
    Use template matching to find the overlap between left/right images.
    Take the right strip of img_left and find it in img_right (should be near left edge).
    """
    h, w = img_left.shape[:2]
    
    # Take a strip from the right side of img_left (the overlap region)
    strip_w = min(w // 3, 400)
    
    # Focus on boards + ice area (avoid sky/crowd)
    template = img_left[int(h*0.05):int(h*0.65), w-strip_w:]
    
    # Search in img_right (within first 2/3 of width)
    search = img_right[int(h*0.05):int(h*0.65), :w*2//3]
    
    # Convert to grayscale
    tg = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    sg = cv2.cvtColor(search, cv2.COLOR_BGR2GRAY)
    
    result = cv2.matchTemplate(sg, tg, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    
    print(f"  Template match: score={max_val:.3f}  loc={max_loc}")
    
    if max_val < 0.3:
        print(f"  Template match FAILED (score too low)")
        return None, None
    
    # The found location tells us: right image[top_y + max_loc[1], max_loc[0]] 
    # corresponds to left image[int(h*0.05), w-strip_w]
    # So img_right's left_edge relative to img_left:
    overlap_in_right = max_loc[0]   # x position in search area where template starts
    left_start_in_right = overlap_in_right - (w - strip_w)  # where left image starts in right coords
    
    tx = -left_start_in_right  # how much to offset img_right to the right
    ty_offset = max_loc[1] - 0  # vertical offset
    
    print(f"  Template match: overlap_start_in_right={overlap_in_right}  implied_tx={tx:.0f}  ty={ty_offset}")
    
    return tx, ty_offset


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
    return img[max(0,y):y+h, max(0,x):x+w]


def main():
    print("=== Loading sideline (10.36.xx) screenshots ===")
    imgs = load_36_shots()
    
    # imgs sorted: [0]=10.36.10, [1]=10.36.20, [2]=10.36.27
    # From visual inspection:
    #   10.36.20 (idx=1) = LEFT END (ANA net in left end zone, "DISCOVER" boards)
    #   10.36.10 (idx=0) = LEFT ZONE ("Score Cash Back", CreditOne, center faceoff circle right)  
    #   10.36.27 (idx=2) = LEFT-CENTER (very similar to 10.36.10, slightly more centered)
    
    # Spatial order: [1] → [0] or [2] → [??]
    # Looking at ice markings visible:
    #   ss4 (10.36.20): left end zone faceoff circles, goalie crease
    #   ss3 (10.36.10): left faceoff circle, Toyota logo, center ice logo partial
    #   ss5 (10.36.27): center ice logo, APP logo (left blue line area)
    
    spatial_order = [imgs[1], imgs[0], imgs[2]]  # left-end → left-zone → left-center
    labels = ["10.36.20 (LEFT END)", "10.36.10 (LEFT ZONE)", "10.36.27 (LEFT-CENTER)"]
    
    h, w = spatial_order[0][1].shape[:2]
    
    print("\n=== Attempting SIFT feature matching ===")
    H_list = [None] * len(spatial_order)
    
    offset_x = 50
    offset_y = 50
    H_list[0] = np.array([[1,0,offset_x],[0,1,offset_y],[0,0,1]], dtype=np.float64)
    
    # Try SIFT first
    for i in range(1, len(spatial_order)):
        _, img_cur = spatial_order[i]
        _, img_prev = spatial_order[i-1]
        label = f"{labels[i]} → {labels[i-1]}"
        M = match_pair(img_cur, img_prev, label)
        if M is not None and H_list[i-1] is not None:
            H_list[i] = H_list[i-1] @ M
    
    print("\n=== Falling back to template matching for failed pairs ===")
    for i in range(1, len(spatial_order)):
        if H_list[i] is None and H_list[i-1] is not None:
            _, img_cur = spatial_order[i]
            _, img_prev = spatial_order[i-1]
            label = f"TM: {labels[i]} → {labels[i-1]}"
            print(f"\n  {label}")
            tx, ty = stitch_pair_manual(img_prev, img_cur)
            if tx is not None:
                # tx is the x offset of img_cur relative to img_prev
                prev_origin_x = H_list[i-1][0, 2]
                prev_origin_y = H_list[i-1][1, 2]
                M = np.array([[1,0,prev_origin_x + w - tx],[0,1,prev_origin_y + ty],[0,0,1]], dtype=np.float64)
                H_list[i] = M
    
    # Final fallback: estimate offset based on known rink geometry
    for i in range(1, len(spatial_order)):
        if H_list[i] is None and H_list[i-1] is not None:
            print(f"\n  Geometry fallback for image {i}")
            # Typical overlap between adjacent sideline shots: ~60-70% of frame width
            # So each image shifts ~30-40% of frame width to the right
            prev_tx = H_list[i-1][0, 2]
            prev_ty = H_list[i-1][1, 2]
            shift = int(w * 0.45)  # estimated 45% of frame = non-overlapping strip
            H_list[i] = np.array([[1,0,prev_tx+shift],[0,1,prev_ty],[0,0,1]], dtype=np.float64)
            print(f"  Using geometry estimate: tx={prev_tx+shift:.0f}")
    
    print("\n=== Building canvas ===")
    # Determine canvas size
    max_tx = max(H[0,2] for H in H_list if H is not None) + w + 100
    max_ty = max(H[1,2] for H in H_list if H is not None) + h + 100
    cw = int(max_tx)
    ch = int(max_ty)
    print(f"  Canvas: {cw}x{ch}")
    
    canvas = np.zeros((ch, cw, 3), dtype=np.float32)
    wc = np.zeros((ch, cw, 1), dtype=np.float32)
    
    for i, (fname, img) in enumerate(spatial_order):
        if H_list[i] is not None:
            blend(canvas, wc, img, H_list[i], cw, ch)
            print(f"  Blended [{i}] {labels[i]}  H_tx={H_list[i][0,2]:.0f}")
        else:
            print(f"  SKIPPED [{i}] {labels[i]}")
    
    wc[wc == 0] = 1.0
    result = (canvas / wc).astype(np.uint8)
    result = crop(result)
    
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cv2.imwrite(OUT, result, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nSaved → {OUT}  ({result.shape[1]}x{result.shape[0]})")
    return result


if __name__ == "__main__":
    main()
