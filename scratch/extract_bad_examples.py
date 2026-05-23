import cv2
import numpy as np
import os
import sys

sys.path.append('src')
from calibration.ml_board_detector import MLBoardDetector
from validate_model import overlay_mask

def process_and_save(detector, img_path, dst_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error loading {img_path}")
        return False
        
    success = detector.detect(img)
    if success:
        mask = detector.get_board_mask()
        prob_map = detector.get_probability_map()
        h, w = img.shape[:2]
        
        # Convert prob_map to 8-bit heatmap for visualization
        prob_heat = np.uint8(prob_map * 255)
        prob_heat = cv2.applyColorMap(prob_heat, cv2.COLORMAP_JET)
        prob_heat = cv2.resize(prob_heat, (w, h))
        
        # Mask visualization
        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Overlay
        overlay = overlay_mask(img.copy(), mask)
        
        # Assemble composite
        top_row = np.hstack((img, overlay))
        bottom_row = np.hstack((prob_heat, mask_vis))
        composite = np.vstack((top_row, bottom_row))
        
        # Add text
        score = detector.get_confidence_score()
        cv2.putText(composite, f"Model: UNET (Challenging Case) | Confidence: {score:.4f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imwrite(dst_path, composite)
        print(f"Successfully processed {img_path} and saved validation to {dst_path}")
        return True
    return False

def main():
    model_path = "src/calibration/board_segmentation_model_unet.pth"
    detector = MLBoardDetector(model_path=model_path)
    
    os.makedirs("images", exist_ok=True)
    
    # 1. Processing Stands Leak Example
    leak_file = "colab_training_data/annotation_frames/new_batch/annotated/2026-05-12_21-12-18_f0352.jpg"
    process_and_save(detector, leak_file, "images/unet_val_bad_1.jpg")
    
    # 2. Processing Low Confidence Example
    low_conf_file = "colab_training_data/annotation_frames/new_batch/unannotated/2026-05-19_23-34-35_f0011.jpg"
    process_and_save(detector, low_conf_file, "images/unet_val_bad_2.jpg")

if __name__ == "__main__":
    main()
