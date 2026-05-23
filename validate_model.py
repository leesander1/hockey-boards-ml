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
    import argparse
    parser = argparse.ArgumentParser(description="Validate ML board detection models")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to specific model weights file (e.g. src/calibration/board_segmentation_model_unet.pth). If None, validates both models.")
    parser.add_argument("--input-dir", type=str, default=None,
                        help="Path to validation images directory. If None, checks several standard locations.")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    if not input_dir:
        # Fallback search order
        candidates = [
            'annotation_frames/new_batch',
            'colab_training_data/annotation_frames/new_batch',
            'scratch/training_data/colab_training_data/annotation_frames/new_batch',
            'test_images/annotation_frames/new_batch'
        ]
        for candidate in candidates:
            if os.path.exists(candidate) and (glob.glob(os.path.join(candidate, '*/*.jpg')) or glob.glob(os.path.join(candidate, '*.jpg'))):
                input_dir = candidate
                break
        if not input_dir:
            input_dir = 'colab_training_data/annotation_frames/new_batch' # default fallback
            
    output_base_dir = 'test_images/validation_output'
    
    # Identify models to validate
    models_to_test = {}
    if args.model:
        name = "custom"
        if "unet" in args.model.lower():
            name = "unet"
        models_to_test[name] = args.model
    else:
        unet_path = os.path.join('src', 'calibration', 'board_segmentation_model.pth')
        if os.path.exists(unet_path):
            models_to_test['unet'] = unet_path
            
    # Get validation images
    files = sorted(glob.glob(os.path.join(input_dir, '*/*.jpg')))
    if not files:
        print(f"Error: No images found in {input_dir}")
        sys.exit(1)
        
    np.random.seed(42)
    test_files = np.random.choice(files, size=min(10, len(files)), replace=False)
    
    for model_name, model_path in models_to_test.items():
        print(f"\n--- 🧪 Validating {model_name.upper()} model ({model_path}) ---")
        detector = MLBoardDetector(model_path=model_path)
        if not detector.is_ready():
            print(f"Error: Model {model_name} could not be loaded. Skipping.")
            continue
            
        output_dir = os.path.join(output_base_dir, model_name)
        os.makedirs(output_dir, exist_ok=True)
        
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
                out_path = os.path.join(output_dir, f"val_{model_name}_{base}")
                
                # Add text
                score = detector.get_confidence_score()
                cv2.putText(composite, f"Model: {model_name.upper()} | Confidence: {score:.3f}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imwrite(out_path, composite)
                print(f"Saved {out_path}")
            else:
                print(f"Failed detection on {f}")

if __name__ == '__main__':
    main()

