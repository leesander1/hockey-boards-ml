import cv2
import numpy as np
import glob

image_files = sorted(glob.glob("src/Screenshot*.png"))
images = [cv2.imread(f) for f in image_files if cv2.imread(f) is not None]

sift = cv2.SIFT_create()

# Image 1 (Right) and Image 2 (Center)
img1 = images[1]
img2 = images[2]

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

h, w = gray1.shape
mask1 = np.zeros_like(gray1)
mask1[int(h*0.2):int(h*0.9), :] = 255  # Exclude very top and bottom

kp1, des1 = sift.detectAndCompute(gray1, mask1)
kp2, des2 = sift.detectAndCompute(gray2, mask1)

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.8 * n.distance:
        good_matches.append(m)

print(f"Good matches: {len(good_matches)}")

src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Estimate affine partial
M_partial, inliers_partial = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
print(f"Partial Affine inliers: {np.sum(inliers_partial)}")

# Estimate full affine
M_full, inliers_full = cv2.estimateAffine2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
print(f"Full Affine inliers: {np.sum(inliers_full)}")

# Estimate homography
H, inliers_H = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
print(f"Homography inliers: {np.sum(inliers_H)}")

# Draw matches for full affine inliers
inlier_matches = [good_matches[i] for i in range(len(good_matches)) if inliers_full[i][0]]
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

output_path = "/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/debug_matches_1_2.jpg"
cv2.imwrite(output_path, img_matches)
print(f"Saved debug matches to {output_path}")

