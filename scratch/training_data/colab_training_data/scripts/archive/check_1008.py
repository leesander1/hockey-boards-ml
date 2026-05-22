import cv2
import glob

files = sorted(glob.glob("src/Screenshot*10.08*.png"))
for f in files:
    img = cv2.imread(f)
    print(f"File: {f.split('/')[-1]}, Size: {img.shape}")
