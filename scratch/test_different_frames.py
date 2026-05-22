import cv2
import numpy as np
import sys
import os
sys.path.append('src')
from calibration.ml_board_detector import MLBoardDetector

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)

def main():
    vid_path = 'data/videos/2026-05-12 21-15-46.mp4'
    out_dir = '/Users/leesander/.gemini/antigravity-ide/brain/19cf1bf7-48df-4651-b795-7722108771ff/different_frames'
    os.makedirs(out_dir, exist_ok=True)
    
    detector = MLBoardDetector()
    if not detector.is_ready():
        print("Model not loaded")
        return
        
    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Failed to open video")
        return
        
    frames_to_test = [100, 250, 400, 550]
    
    for idx in frames_to_test:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {idx}")
            continue
            
        success = detector.detect(frame)
        if success:
            mask = detector.get_board_mask()
            raw_prob = detector.get_probability_map()
            
            # Create a heatmap of the raw probabilities
            h, w = frame.shape[:2]
            prob_heat = np.uint8(raw_prob * 255)
            prob_heat = cv2.applyColorMap(prob_heat, cv2.COLORMAP_JET)
            prob_heat = cv2.resize(prob_heat, (w, h))
            
            # Overlay final mask
            overlay = overlay_mask(frame.copy(), mask)
            
            # Stack horizontally: Original, Raw Prob Map, Final Mask Overlay
            h_stack = np.hstack((frame, prob_heat, overlay))
            
            out_file = os.path.join(out_dir, f"frame_{idx:03d}.jpg")
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(h_stack, "Original", (10, 40), font, 1, (255, 255, 255), 2)
            cv2.putText(h_stack, "Raw UNet Prob Map", (w + 10, 40), font, 1, (255, 255, 255), 2)
            cv2.putText(h_stack, "Post-Processed Mask", (w*2 + 10, 40), font, 1, (255, 255, 255), 2)
            
            cv2.imwrite(out_file, h_stack)
            print(f"Saved {out_file}")
            
    cap.release()

if __name__ == "__main__":
    main()
