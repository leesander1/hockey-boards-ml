import cv2
import numpy as np
import glob

image_files = sorted(glob.glob("src/Screenshot*.png"))
images = [cv2.imread(f) for f in image_files]

sift = cv2.SIFT_create()
features = []

for i, img in enumerate(images):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = np.zeros_like(gray)
    mask[int(h*0.2):int(h*0.6), :] = 255
    kp, des = sift.detectAndCompute(gray, mask)
    features.append((kp, des))
    print(f"Image {i} ({image_files[i].split('/')[-1]}): {len(kp)} features")

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

for i in range(3):
    for j in range(i+1, 3):
        kp1, des1 = features[i]
        kp2, des2 = features[j]
        
        matches = flann.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
                
        print(f"Matches between {i} and {j}: {len(good_matches)}")

