import cv2
import numpy as np
import glob

def main():
    image_files = sorted(glob.glob("src/Screenshot*.png"))
    # 1: 10.08.46 PM
    # 2: 10.08.49 PM
    img1 = cv2.imread(image_files[1])
    img2 = cv2.imread(image_files[2])

    sift = cv2.SIFT_create()
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    h, w = gray1.shape
    mask = np.zeros_like(gray1)
    mask[int(h*0.1):int(h*0.95), :] = 255 

    kp1, des1 = sift.detectAndCompute(gray1, mask)
    kp2, des2 = sift.detectAndCompute(gray2, mask)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des2, des1, k=2)  # Match img2 to img1
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    print(f"Good matches 2 -> 1: {len(good_matches)}")
    
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)

    canvas_w = int(w * 3)
    canvas_h = int(h * 2)
    offset_x = w
    offset_y = h // 2
    
    H_base = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    result = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Draw img1 (center)
    warped_img1 = cv2.warpPerspective(img1, H_base, (canvas_w, canvas_h))
    
    # Draw img2 (right)
    H = np.matmul(H_base, M)
    warped_img2 = cv2.warpPerspective(img2, H, (canvas_w, canvas_h))

    # Blend them: draw img2 first, then img1 on top? 
    # Or draw img1 first, then img2 on top! That way img2 (which has the new right side) is fully visible!
    mask1 = cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0
    result[mask1] = warped_img1[mask1]

    mask2 = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0
    # Only overwrite where result is empty, or just overwrite entirely?
    # Overwrite entirely!
    result[mask2] = warped_img2[mask2]

    # Crop
    gray_res = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_res, 1, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, crop_w, crop_h = cv2.boundingRect(coords)
        result = result[y:y+crop_h, x:x+crop_w]

    output_path = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/stitched_2_photos_1_and_2.jpg"
    cv2.imwrite(output_path, result)
    print(f"Saved 2-photo stitch to {output_path}")

if __name__ == "__main__":
    main()
