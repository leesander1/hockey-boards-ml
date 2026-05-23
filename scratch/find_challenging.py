import cv2
import numpy as np
import glob
import os
import sys

sys.path.append('src')
from calibration.ml_board_detector import MLBoardDetector

def main():
    # Load detector
    model_path = "src/calibration/board_segmentation_model_unet.pth"
    detector = MLBoardDetector(model_path=model_path)
    
    # Path to validation files
    input_dir = 'colab_training_data/annotation_frames/new_batch'
    files = sorted(glob.glob(os.path.join(input_dir, '*/*.jpg')))
    print(f"Analyzing {len(files)} files...")
    
    results = []
    
    for f in files:
        img = cv2.imread(f)
        if img is None: continue
        
        success = detector.detect(img)
        if not success: continue
        
        mask = detector.get_board_mask()
        prob_map = detector.get_probability_map()
        h, w = img.shape[:2]
        
        # 1. Check for leak in stands (top 25% of the frame)
        stands_region = mask[0:int(h*0.25), :]
        stands_leak_px = np.sum(stands_region > 0)
        stands_leak_pct = stands_leak_px / stands_region.size
        
        # 2. Check for leak in bottom area (camera housing / crowd, bottom 15%)
        bottom_region = mask[int(h*0.85):, :]
        bottom_leak_px = np.sum(bottom_region > 0)
        bottom_leak_pct = bottom_leak_px / bottom_region.size
        
        # 3. Overall confidence (percentiles of prob map)
        p99 = np.percentile(prob_map, 99)
        p50 = np.percentile(prob_map, 50)
        
        results.append({
            'file': f,
            'stands_leak_pct': stands_leak_pct,
            'bottom_leak_pct': bottom_leak_pct,
            'p99': p99,
            'p50': p50,
            'score': detector.get_confidence_score()
        })
        
    # Sort by stands leak percentage (highest first) to find leaking frames
    print("\n--- 🚨 Top 5 Frames with Stands Leaks (Challenging Cases) ---")
    results_sorted_leak = sorted(results, key=lambda x: x['stands_leak_pct'], reverse=True)
    for r in results_sorted_leak[:5]:
        print(f"File: {r['file']} | Stands Leak: {r['stands_leak_pct']*100:.2f}% | Conf Score: {r['score']:.4f}")
        
    # Sort by lowest confidence score
    print("\n--- 📉 Top 5 Lowest Confidence Score Frames ---")
    results_sorted_conf = sorted(results, key=lambda x: x['score'])
    for r in results_sorted_conf[:5]:
        print(f"File: {r['file']} | Conf Score: {r['score']:.4f} | Stands Leak: {r['stands_leak_pct']*100:.2f}%")

if __name__ == "__main__":
    main()
