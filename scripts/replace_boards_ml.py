#!/usr/bin/env python3
"""
Pipeline script to replace real board advertisements in a video
using the trained ML board segmentation model and YOLOv8 player segmentation.
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
    parser = argparse.ArgumentParser(description="Replace boards in a video using the trained ML model")
    parser.add_argument("--video", type=str, default="data/videos/2026-05-12 21-15-46.mp4", 
                        help="Path to source video")
    parser.add_argument("--ad", type=str, default="test_images/premium_ad.png", 
                        help="Path to replacement advertisement banner")
    parser.add_argument("--output", type=str, default="output/ml_composited.mp4", 
                        help="Path to save the output video")
    parser.add_argument("--max-frames", type=int, default=150, 
                        help="Maximum frames to process (default: 150 for a fast demo)")
    
    args = parser.parse_args()
    
    # Check paths
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        # Look for fallback video
        video_dir = "data/videos"
        if os.path.exists(video_dir):
            vids = [os.path.join(video_dir, v) for v in os.listdir(video_dir) if v.endswith(".mp4")]
            if vids:
                print(f"Using fallback video: {vids[0]}")
                args.video = vids[0]
            else:
                sys.exit(1)
        else:
            sys.exit(1)
            
    if not os.path.exists(args.ad):
        print(f"Warning: Ad banner not found at {args.ad}. Using green fallback.")
        args.ad = None

    # Ensure output dir exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print("\n--- 🚀 Starting ML Board Replacement Pipeline ---")
    print(f"Source Video  : {args.video}")
    print(f"Replacement Ad: {args.ad}")
    print(f"Output Video  : {args.output}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")
    
    # 1. Load ML Board Detector (our trained UNet)
    model_path = os.path.join("src", "calibration", "board_segmentation_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}!")
        sys.exit(1)
        
    print("Loading ML Board Detector...")
    board_detector = MLBoardDetector(model_path=model_path)
    if not board_detector.is_ready():
        print("Error: Failed to initialize ML Board Detector.")
        sys.exit(1)
        
    # 2. Load Model Runner for Player Segmentation
    print("Loading YOLOv8 Player Segmentation...")
    runner = ModelRunner(player_model_path="yolov8n-seg.pt", device=device)
    
    # 3. Load Ad Compositor
    compositor = AdCompositor(ad_image_path=args.ad)
    
    # 4. Open Video Stream
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video}")
        sys.exit(1)
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 29.97
        
    limit_frames = min(args.max_frames, total_frames) if args.max_frames > 0 else total_frames
    
    print(f"Video specs: {width}x{height} | {fps:.2f} FPS | {total_frames} total frames")
    print(f"Processing limit: {limit_frames} frames")
    
    # 5. Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # 6. Processing Loop
    frames_processed = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # A. Detect boards (ML Board Segmentation Model)
            success = board_detector.detect(frame)
            if success:
                board_mask = board_detector.get_board_mask()
            else:
                board_mask = np.zeros((height, width), dtype=np.uint8)
                
            # B. Detect players (YOLOv8 instance segmentation)
            player_mask = runner.get_player_mask(frame)
            
            # C. Blend ad behind players inside board area
            composited = compositor.apply_ad(frame, board_mask, player_mask)
            
            # D. Write to output
            out.write(composited)
            
            frames_processed += 1
            if frames_processed % 10 == 0 or frames_processed == 1:
                elapsed = time.time() - start_time
                fps_processing = frames_processed / elapsed
                print(f"Processed {frames_processed}/{limit_frames} frames ({fps_processing:.1f} FPS)...")
                
            if args.max_frames > 0 and frames_processed >= args.max_frames:
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving progress...")
    finally:
        cap.release()
        out.release()
        
    elapsed_total = time.time() - start_time
    print(f"\n--- Output saved successfully to {args.output} ---")
    print(f"Processed {frames_processed} frames in {elapsed_total:.1f} seconds ({frames_processed/elapsed_total:.1f} FPS average)")
    print("---------------------------------------------------\n")

if __name__ == "__main__":
    main()
