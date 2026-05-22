#!/usr/bin/env python3
"""Densely generate high-confidence pseudo-labels directly from raw videos using the trained U-Net."""

import cv2
import os
import glob
import numpy as np
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.calibration.ml_board_detector import MLBoardDetector

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)

def main():
    detector = MLBoardDetector()
    if not detector.is_ready():
        print("Error: Could not load MLBoardDetector model.")
        return

    output_dir = 'annotation_frames/pseudo_labeled'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clean up old files in the directory to prevent duplicate or stale data
    print("Clearing old pseudo-labeled files from annotation_frames/pseudo_labeled/...")
    old_files = glob.glob(os.path.join(output_dir, '*.jpg')) + glob.glob(os.path.join(output_dir, '*.png'))
    for f in old_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"Warning: Failed to remove {f}: {e}")

    videos = sorted(glob.glob('data/videos/*.mp4'))
    if not videos:
        print("Error: No videos found in data/videos/.")
        return
        
    print(f"Found {len(videos)} videos in data/videos/. Starting dense pseudo-labeling...")
    
    sample_interval = 30  # sample 1 frame per second at 30 fps
    count_saved = 0
    count_skipped = 0

    for idx, video_path in enumerate(videos):
        vid_name = os.path.basename(video_path).split('.')[0].replace(' ', '_')
        print(f"\n[{idx+1}/{len(videos)}] Processing: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}. Skipping.")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            print(f"Warning: {video_path} has 0 frames. Skipping.")
            cap.release()
            continue
            
        print(f"  Streaming {total_frames} frames... Seeking every {sample_interval} frames.")
        
        for frame_idx in range(0, total_frames, sample_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break
                
            success = detector.detect(frame)
            if success:
                mask = detector.get_board_mask()
                conf = detector.get_confidence_score()
                board_ratio = (mask > 0).mean()
                
                # Strict high-confidence filtering for self-training stability
                if conf > 0.85 and 0.02 < board_ratio < 0.25:
                    stem = f"{vid_name}_f{frame_idx:05d}"
                    
                    # Save raw frame
                    cv2.imwrite(os.path.join(output_dir, f"{stem}.jpg"), frame)
                    
                    # Save straightened mask
                    cv2.imwrite(os.path.join(output_dir, f"{stem}-mask.png"), mask)
                    
                    # Save overlay for QA inspection
                    overlay = overlay_mask(frame, mask)
                    cv2.imwrite(os.path.join(output_dir, f"{stem}-annotated.png"), overlay)
                    
                    count_saved += 1
                else:
                    count_skipped += 1
                    
        cap.release()
        print(f"  Finished {vid_name}. (Saved {count_saved} total so far, skipped {count_skipped})")

    print(f"\nDone! Successfully extracted {count_saved} pseudo-labeled training pairs. Skipped {count_skipped} low-confidence frames.")

if __name__ == '__main__':
    main()
