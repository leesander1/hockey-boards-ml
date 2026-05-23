import cv2
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO

sys.path.append('src')
from calibration.ml_board_detector import MLBoardDetector
from inference.model_runner import ModelRunner
from compositing.homography import AdCompositor

def guided_filter(I, p, r, eps):
    # I: grayscale guide image [0, 1]
    # p: binary mask [0, 1]
    N = cv2.boxFilter(np.ones_like(I), -1, (2*r+1, 2*r+1))
    
    mean_I = cv2.boxFilter(I, -1, (2*r+1, 2*r+1)) / N
    mean_p = cv2.boxFilter(p, -1, (2*r+1, 2*r+1)) / N
    mean_Ip = cv2.boxFilter(I * p, -1, (2*r+1, 2*r+1)) / N
    
    cov_Ip = mean_Ip - mean_I * mean_p
    
    mean_II = cv2.boxFilter(I * I, -1, (2*r+1, 2*r+1)) / N
    var_I = mean_II - mean_I * mean_I
    
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    
    mean_a = cv2.boxFilter(a, -1, (2*r+1, 2*r+1)) / N
    mean_b = cv2.boxFilter(b, -1, (2*r+1, 2*r+1)) / N
    
    q = mean_a * I + mean_b
    return np.clip(q, 0.0, 1.0)

def main():
    vid_path = 'data/videos/2026-05-19 23-34-03.mp4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    runner = ModelRunner(player_model_path="yolov8m-seg.pt", device=device)
    cap = cv2.VideoCapture(vid_path)
    
    # Grab frame 50
    cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Failed to read frame")
        return
        
    h, w = frame.shape[:2]
    
    # Get raw, non-dilated mask
    raw_mask = runner.get_player_mask(frame, dilation_kernel_size=0)
    
    # Apply Guided Filter
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p = raw_mask.astype(np.float32) / 255.0
    
    refined_mask = guided_filter(gray_frame, p, r=4, eps=0.01)
    
    # Save the comparison
    # Draw original frame, raw YOLO mask overlay, and refined mask overlay
    raw_overlay = frame.copy()
    raw_overlay[raw_mask > 0] = [0, 0, 255] # Red
    
    refined_overlay = frame.copy()
    refined_mask_uint8 = (refined_mask * 255).astype(np.uint8)
    refined_overlay[refined_mask_uint8 > 127] = [0, 255, 0] # Green
    
    # Zoom in on a player overlapping the boards to see details
    # We can crop a region that contains the player near the boards
    # Let's find contours in raw_mask to locate a player
    contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find one overlapping the boards (in the vertical center/bottom)
        best_contour = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(best_contour)
        
        # Expand crop box slightly
        pad = 40
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + w_box + pad)
        y2 = min(h, y + h_box + pad)
        
        crop_orig = frame[y1:y2, x1:x2]
        crop_raw = raw_overlay[y1:y2, x1:x2]
        crop_refined = refined_overlay[y1:y2, x1:x2]
        
        stacked = np.hstack((crop_orig, crop_raw, crop_refined))
        cv2.imwrite('scratch/guided_comparison.jpg', stacked)
        print("Saved scratch/guided_comparison.jpg")
    else:
        print("No players found in frame")

if __name__ == "__main__":
    main()
