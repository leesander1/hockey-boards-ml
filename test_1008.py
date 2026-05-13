import cv2
import numpy as np

for f in ["src/Screenshot 2026-05-12 at 10.08.20 PM.png", 
          "src/Screenshot 2026-05-12 at 10.08.46 PM.png", 
          "src/Screenshot 2026-05-12 at 10.08.49 PM.png"]:
    img = cv2.imread(f)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Check for center ice
    print(f"{f}: mean brightness={np.mean(gray):.1f}")
