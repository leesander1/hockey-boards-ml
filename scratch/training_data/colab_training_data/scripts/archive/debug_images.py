import cv2
import numpy as np
import glob

image_files = sorted(glob.glob("src/Screenshot*.png"))
images = [cv2.imread(f) for f in image_files if cv2.imread(f) is not None]

for i, f in enumerate(image_files):
    name = f.split('/')[-1].replace(' ', '_').replace(':', '_')
    out_path = f"/Users/leesander/.gemini/antigravity/brain/d5aae78c-4b02-47aa-b503-4927926f04c0/artifacts/debug_orig_{name}"
    cv2.imwrite(out_path, images[i])
    print(f"Saved {out_path}")
