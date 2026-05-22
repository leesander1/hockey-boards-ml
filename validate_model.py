import cv2
import os
import glob
import numpy as np
import sys

# Ensure src is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from calibration.ml_board_detector import MLBoardDetector

def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlays a binary mask on an image with the given color and opacity."""
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    return cv2.addWeighted(image, 1.0, colored_mask, alpha, 0)

def main():
    detector = MLBoardDetector()
    
    input_dir = 'annotation_frames/new_batch'
    output_dir = 'test_images/validation_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a mix of regular and newly annotated images
    files = sorted(glob.glob(os.path.join(input_dir, '*/*.jpg')))
    
    # Let's take just a few to test
    np.random.seed(42)
    test_files = np.random.choice(files, size=min(10, len(files)), replace=False)
    
    print(f"Running validation on {len(test_files)} images...")
    for idx, f in enumerate(test_files):
        img = cv2.imread(f)
        if img is None:
            continue
            
        success = detector.detect(img)
        if success:
            mask = detector.get_board_mask()
            prob_map = detector.get_probability_map()
            
            # Create a composite image: original, prob map, thresholded mask, overlay
            h, w = img.shape[:2]
            
            # Convert prob_map to 8-bit heatmap for visualization
            prob_heat = np.uint8(prob_map * 255)
            prob_heat = cv2.applyColorMap(prob_heat, cv2.COLORMAP_JET)
            prob_heat = cv2.resize(prob_heat, (w, h))
            
            # Mask visualization
            mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            
            # Overlay
            overlay = overlay_mask(img.copy(), mask)
            
            # Top row: original, overlay
            # Bottom row: prob map, mask
            top_row = np.hstack((img, overlay))
            bottom_row = np.hstack((prob_heat, mask_vis))
            composite = np.vstack((top_row, bottom_row))
            
            # Save
            base = os.path.basename(f)
            out_path = os.path.join(output_dir, f"val_v3_{base}")
            
            # Add text
            score = detector.get_confidence_score()
            cv2.putText(composite, f"Confidence: {score:.3f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite(out_path, composite)
            print(f"Saved {out_path}")
        else:
            print(f"Failed detection on {f}")

if __name__ == '__main__':
    main()
