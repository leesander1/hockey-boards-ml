#!/usr/bin/env python3
"""
Real-Time Interactive Hockey Ad Blocker & Board Replacer.
Streams a video (or live webcam/capture card) and displays the ad-blocked output live on screen.
Fully optimized for Apple Silicon (MPS) GPU acceleration.
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.calibration.ml_board_detector import MLBoardDetector
from src.inference.model_runner import ModelRunner
from src.compositing.homography import AdCompositor

def main():
    parser = argparse.ArgumentParser(description="Real-Time ML Rink Board Ad Blocker")
    parser.add_argument("--source", type=str, default="data/videos/2026-05-12 21-15-46.mp4", 
                        help="Path to video file or '0' for live webcam/capture card")
    parser.add_argument("--ad", type=str, default="test_images/neutral_board.png", 
                        help="Path to replacement board texture")
    parser.add_argument("--blend-alpha", type=float, default=0.70, 
                        help="Blending alpha: 1.0 = fully opaque new texture, 0.0 = original boards")
    parser.add_argument("--model", type=str, default=os.path.join("src", "calibration", "board_segmentation_model.pth"),
                        help="Path to trained model weights (.pth)")
    
    args = parser.parse_args()
    
    # Handle live source vs file
    is_live = False
    if args.source.isdigit():
        args.source = int(args.source)
        is_live = True
        print(f"Using live video source: Camera index {args.source}")
    else:
        if not os.path.exists(args.source):
            print(f"Error: Video file not found: {args.source}")
            # Try to search for any video inside data/videos
            video_dir = "data/videos"
            if os.path.exists(video_dir):
                vids = [os.path.join(video_dir, v) for v in os.listdir(video_dir) if v.endswith(".mp4")]
                if vids:
                    print(f"Using fallback video file: {vids[0]}")
                    args.source = vids[0]
                else:
                    sys.exit(1)
            else:
                sys.exit(1)
                
    if not os.path.exists(args.ad):
        print(f"Warning: Ad banner not found at {args.ad}. Using fallback.")
        args.ad = None

    if not os.path.exists(args.model):
        print(f"Error: Model weights not found at {args.model}")
        sys.exit(1)

    # 1. Device selection - support Apple Silicon GPU (MPS) & NVIDIA GPU (CUDA)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\n--- 🚀 Initializing Real-Time Interactive Ad Blocker ---")
    print(f"Hardware Acceleration: {device.type.upper()}")
    print(f"Video Source         : {args.source}")
    print(f"Model Path           : {args.model}")
    print(f"Replacement Ad       : {args.ad}")
    print(f"-------------------------------------------------------")

    # 2. Initialize modules
    print("Loading ML Board Segmenter...")
    board_detector = MLBoardDetector(model_path=args.model)
    if not board_detector.is_ready():
        print("Error: Failed to initialize ML Board Detector.")
        sys.exit(1)
        
    print("Loading YOLOv8 Player Occlusion Model...")
    runner = ModelRunner(player_model_path="yolov8n-seg.pt", device=device)
    
    compositor = AdCompositor(ad_image_path=args.ad)

    # 3. Open Video stream
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Cannot open video source: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    print(f"\nSuccessfully opened stream: {width}x{height} @ {fps:.1f} FPS")
    print("\n[Controls]")
    print("  - Press 's' to TOGGLE ad blocking on / off")
    print("  - Press 'q' to QUIT the live demo")
    print("------------------------------------------")

    blocking_enabled = True
    fps_history = []
    
    cv2.namedWindow("Real-Time Hockey Ad Blocker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real-Time Hockey Ad Blocker", 1280, 720)

    try:
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                if is_live:
                    continue
                else:
                    print("End of video stream. Restarting playback...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

            # Process frame
            if blocking_enabled:
                # A. Segment rink boards
                board_detector.detect(frame)
                board_mask = board_detector.get_board_mask()

                # B. Segment foreground players for occlusion preservation
                player_mask = runner.get_player_mask(frame)

                # C. Overlay replacement ad behind players
                display_frame = compositor.apply_ad(frame, board_mask, player_mask, blend_alpha=args.blend_alpha)
            else:
                display_frame = frame.copy()

            # Calculate actual processing FPS
            t_elapsed = time.time() - t_start
            current_fps = 1.0 / max(t_elapsed, 0.001)
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Draw sleek HUD overlay on frame
            status_text = "AD BLOCKER: ACTIVE" if blocking_enabled else "AD BLOCKER: BYPASSED"
            status_color = (0, 255, 0) if blocking_enabled else (0, 0, 255)
            
            # semi-transparent HUD background
            cv2.rectangle(display_frame, (10, 10), (370, 110), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, 10), (370, 110), status_color, 2)
            
            # HUD text details
            cv2.putText(display_frame, status_text, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_frame, f"Device: {device.type.upper()}", (25, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"Performance: {avg_fps:.1f} FPS", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Display frame live
            cv2.imshow("Real-Time Hockey Ad Blocker", display_frame)

            # Interactive keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                blocking_enabled = not blocking_enabled
                print(f"Ad-blocker state toggled to: {'ACTIVE' if blocking_enabled else 'BYPASSED'}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nDemo finished. Closed stream.")

if __name__ == '__main__':
    main()
