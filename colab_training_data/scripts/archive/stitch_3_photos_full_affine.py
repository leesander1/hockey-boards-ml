import cv2
import numpy as np
import glob

def main():
    image_files = sorted(glob.glob("src/Screenshot*.png"))
    if len(image_files) != 3: return
    images = [cv2.imread(f) for f in image_files if cv2.imread(f) is not None]

    sift = cv2.SIFT_create()
    features = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.zeros_like(gray)
        mask[int(h*0.1):int(h*0.95), :] = 255 
        kp, des = sift.detectAndCompute(gray, mask)
        features.append((kp, des))

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    center_idx = 2
    h, w = images[center_idx].shape[:2]
    canvas_w = int(w * 4)
    canvas_h = int(h * 3)
    offset_x = int(w * 1.5)
    offset_y = h
    
    H_base = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    def get_homography(idx_src, idx_dst):
        kp1, des1 = features[idx_src]
        kp2, des2 = features[idx_dst]
        
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good_matches.append(m)
                
        if len(good_matches) < 10: return None
        
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Use Full Affine
        M_aff, inliers = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        M = np.eye(3, dtype=np.float32)
        M[0:2, :] = M_aff
        return M

    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    warped_img_c = cv2.warpPerspective(images[center_idx], H_base, (canvas_w, canvas_h))
    
    warped_imgs = []
    for i in range(3):
        if i == center_idx: continue
        M = get_homography(i, center_idx)
        if M is not None:
            H = np.matmul(H_base, M)
            w_img = cv2.warpPerspective(images[i], H, (canvas_w, canvas_h))
            warped_imgs.append(w_img)

    for w_img in warped_imgs:
        mask = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY) > 0
        result[mask] = w_img[mask]
        
    mask_c = cv2.cvtColor(warped_img_c, cv2.COLOR_BGR2GRAY) > 0
    result[mask_c] = warped_img_c[mask_c]

    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_res, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, crop_w, crop_h = cv2.boundingRect(coords)
        result = result[y:y+crop_h, x:x+crop_w]

    output_path = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_3_photos_full_affine.jpg"
    cv2.imwrite(output_path, result)
    print(f"Saved full affine 3-photo stitch to {output_path}")

if __name__ == "__main__":
    main()
