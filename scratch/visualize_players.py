import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

def main():
    video_path = "data/videos/2026-05-12 21-15-46.mp4"
    output_dir = "scratch/player_viz"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Loading YOLO model on {device}...")
    model = YOLO('yolov8n-seg.pt')
    model.to(device)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return
        
    # Get 3 frames: e.g. frame 10, 50, 90
    frame_indices = [10, 50, 90]
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Could not read frame {idx}")
            continue
            
        # Run YOLO inference
        results = model(frame, verbose=False)
        result = results[0]
        
        # Create an empty mask for all players
        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        if result.masks is not None:
            # We specifically look for "person" class (class 0 in COCO) or just take all for this viz
            for i, mask in enumerate(result.masks.data):
                class_id = int(result.boxes.cls[i].item())
                if class_id == 0: # 0 is person
                    # Mask is typically (H, W) or resized.
                    # Convert to numpy and resize to match image
                    m = mask.cpu().numpy()
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    combined_mask[m > 0] = 255
                    
        # Visualization 1: The player mask overlay (Red for players)
        player_overlay = frame.copy()
        player_overlay[combined_mask > 0] = [0, 0, 255] # Red BGR
        # blend
        cv2.addWeighted(player_overlay, 0.6, frame, 0.4, 0, player_overlay)
        
        # Visualization 2: What is "Subtracted" (The board mask with players cut out)
        # For context, let's just show a fake board mask (e.g. middle of screen) and subtract the players
        fake_board_mask = np.zeros((h, w), dtype=np.uint8)
        fake_board_mask[int(h*0.3):int(h*0.7), :] = 255
        
        final_board_mask = cv2.bitwise_and(fake_board_mask, cv2.bitwise_not(combined_mask))
        
        # Colorize the final board mask green
        subtracted_viz = frame.copy()
        subtracted_viz[final_board_mask > 0] = [0, 255, 0] # Green BGR
        cv2.addWeighted(subtracted_viz, 0.5, frame, 0.5, 0, subtracted_viz)
        
        # Stack images horizontally: Original, Player Detection, Subtracted Result
        h_stack = np.hstack((frame, player_overlay, subtracted_viz))
        
        out_path = os.path.join(output_dir, f"frame_{idx:03d}.jpg")
        
        # add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(h_stack, "Original", (10, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(h_stack, "YOLO Players Detected", (w + 10, 40), font, 1, (255, 255, 255), 2)
        cv2.putText(h_stack, "Board Ad Overlay (Players Subtracted)", (w*2 + 10, 40), font, 1, (255, 255, 255), 2)
        
        cv2.imwrite(out_path, h_stack)
        print(f"Saved {out_path}")
        
    cap.release()

if __name__ == "__main__":
    main()
