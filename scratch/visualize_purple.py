import cv2
import numpy as np
import sys
import os
import torch
from ultralytics import YOLO

sys.path.append('src')
from calibration.ml_board_detector import MLBoardDetector

def main():
    vid_path = 'data/videos/2026-05-12 21-12-18.mp4'
    out_dir = '/Users/leesander/.gemini/antigravity-ide/brain/19cf1bf7-48df-4651-b795-7722108771ff/purple_viz'
    os.makedirs(out_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    detector = MLBoardDetector()
    yolo_model = YOLO('yolov8m-seg.pt')
    yolo_model.to(device)
    
    cap = cv2.VideoCapture(vid_path)
    frames_to_test = [100, 250, 400, 550]
    
    for idx in frames_to_test:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        h, w = frame.shape[:2]
        
        # 1. UNet Board Mask
        success = detector.detect(frame)
        board_mask = detector.get_board_mask() if success else np.zeros((h, w), dtype=np.uint8)
        
        # 2. YOLO Player Mask
        results = yolo_model(frame, verbose=False)
        result = results[0]
        player_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            for i, mask in enumerate(result.masks.data):
                class_id = int(result.boxes.cls[i].item())
                if class_id == 0:
                    m = mask.cpu().numpy()
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    player_mask[m > 0] = 255
            # No dilation to match model_runner's raw YOLO output refined by Guided Filter
            pass
                    
        # 3. Create Visualization
        viz = frame.copy()
        
        # Draw Boards in Green
        board_overlay = np.zeros_like(viz)
        board_overlay[board_mask > 0] = [0, 255, 0] # Green
        
        # Draw Players in Purple (BGR: 255, 0, 255)
        player_overlay = np.zeros_like(viz)
        player_overlay[player_mask > 0] = [255, 0, 255] # Purple/Magenta
        
        # Subtracted final board area (where ad actually goes)
        final_ad_area = cv2.bitwise_and(board_mask, cv2.bitwise_not(player_mask))
        final_overlay = np.zeros_like(viz)
        final_overlay[final_ad_area > 0] = [0, 255, 0]
        
        # Blend
        viz1 = cv2.addWeighted(viz, 0.7, board_overlay, 0.3, 0)
        viz1 = cv2.addWeighted(viz1, 1.0, player_overlay, 0.5, 0)
        
        viz2 = cv2.addWeighted(viz, 0.5, final_overlay, 0.5, 0)
        
        # Stack
        h_stack = np.hstack((viz1, viz2))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(h_stack, "Raw Board (Green) + YOLO Players (Purple)", (10, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(h_stack, "Final Subtracted Ad Area", (w + 10, 40), font, 1, (255, 255, 255), 2)
        
        out_file = os.path.join(out_dir, f"frame_{idx:03d}.jpg")
        cv2.imwrite(out_file, h_stack)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    main()
