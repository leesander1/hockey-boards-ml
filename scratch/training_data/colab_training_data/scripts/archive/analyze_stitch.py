import cv2
import numpy as np

img = cv2.imread("src/stitch.png")
if img is not None:
    print(f"stitch.png: {img.shape}")
else:
    print("Could not load stitch.png")
