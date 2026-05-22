import cv2
import numpy as np
import sys
sys.path.append('src')
from calibration.ml_board_detector import MLBoardDetector
import os

os.makedirs('scratch/debug', exist_ok=True)
detector = MLBoardDetector()
img = cv2.imread('test_images/test_1.jpg') # Hope this exists
if img is None:
    img = cv2.imread('test_images/boards.png')
if img is None:
    # Just grab a frame from the video
    cap = cv2.VideoCapture('data/videos/2026-05-12 21-15-46.mp4')
    cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
    ret, img = cap.read()
    
detector.detect(img)
prob_map = detector.get_probability_map()
cv2.imwrite('scratch/debug/prob_map.png', (prob_map * 255).astype(np.uint8))
cv2.imwrite('scratch/debug/orig.jpg', img)
cv2.imwrite('scratch/debug/final_mask.png', detector.get_board_mask())
