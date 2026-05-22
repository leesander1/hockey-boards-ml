#!/usr/bin/env python3
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
    model_path = 'src/calibration/board_segmentation_model_unet.pth'
    if not os.path.exists(model_path):
        print(f"Error: {model_path} does not exist!")
        return

    print("Initializing MLBoardDetector with U-Net weights...")
    detector = MLBoardDetector(model_path=model_path)
    if not detector.is_ready():
        print("Error: Detector is not ready!")
        return

    input_dir = 'annotation_frames/new_batch/unannotated'
    output_dir = 'annotation_frames/pseudo_labeled_recovered'
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear output directory if any files exist from previous test runs
    files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
    print(f"Found {len(files)} unannotated frames in {input_dir}. Generating recovered pseudo-labels...")
    
    count_saved = 0
    count_skipped = 0

    for idx, f in enumerate(files):
        img = cv2.imread(f)
        if img is None: continue
            
        success = detector.detect(img)
        if success:
            mask = detector.get_board_mask()
            conf = detector.get_confidence_score()
            board_ratio = (mask > 0).mean()
            
            # Strict filtering for high quality (exactly matches old script's threshold of 0.70)
            if conf > 0.70 and 0.02 < board_ratio < 0.25:
                base_name = os.path.basename(f)
                name_no_ext = os.path.splitext(base_name)[0]
                
                # Copy original image
                orig_out = os.path.join(output_dir, base_name)
                shutil.copy2(f, orig_out)
                
                # Save mask as 8-bit grayscale (0 or 255)
                mask_out = os.path.join(output_dir, f"{name_no_ext}-mask.png")
                cv2.imwrite(mask_out, mask)
                
                # Save annotated overlay for visual inspection
                overlay = overlay_mask(img, mask)
                overlay_out = os.path.join(output_dir, f"{name_no_ext}-annotated.png")
                cv2.imwrite(overlay_out, overlay)
                
                count_saved += 1
            else:
                count_skipped += 1
                
        if (idx + 1) % 25 == 0 or idx == len(files) - 1:
            print(f"Processed {idx + 1}/{len(files)}... (saved {count_saved}, last conf {conf:.2f}, ratio {board_ratio:.2f})")

    print(f"Done! Successfully recovered {count_saved} pseudo-labeled pairs into {output_dir}. Skipped {count_skipped} low-confidence frames.")

if __name__ == '__main__':
    main()
