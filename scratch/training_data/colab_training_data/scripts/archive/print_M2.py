import cv2
import numpy as np
import glob

image_files = sorted(glob.glob("src/Screenshot*.png"))
images = [cv2.imread(f) for f in image_files if cv2.imread(f) is not None]

sift = cv2.SIFT_create()
features = []

for img in images:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = np.zeros_like(gray)
    mask[int(h*0.2):int(h*0.7), :] = 255 
    kp, des = sift.detectAndCompute(gray, mask)
    features.append((kp, des))

flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))

def get_homography(idx_src, idx_dst):
    kp1, des1 = features[idx_src]
    kp2, des2 = features[idx_dst]
    
    matches = flann.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)
            
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
    print(f"Mask 0.2-0.7 | Matches {idx_src}->{idx_dst}: {len(good_matches)}, Inliers: {np.sum(inliers)}")

get_homography(0, 1)
get_homography(2, 1)

