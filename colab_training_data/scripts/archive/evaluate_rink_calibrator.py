import cv2
import os
import glob
import numpy as np
from src.calibration.rink_calibrator import RinkCalibrator
from src.calibration.annotation_board_detector import AnnotationBoardDetector

def calculate_coverage_and_bbox(mask):
    if mask is None or np.sum(mask) == 0:
        return 0, (0, 0, 0, 0)
    
    # Calculate coverage
    coverage = np.sum(mask > 0) / mask.size
    
    # Calculate bbox (x, y, w, h)
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    return coverage, (x, y, w, h)

def main():
    frames = sorted(glob.glob("annotation_frames/*.jpg"))
    os.makedirs("tmp_frames", exist_ok=True)
    
    detector = AnnotationBoardDetector()
    calibrator = RinkCalibrator()

    print(f"{'Filename':<40} | {'Detector':<8} | {'Cov %':<6} | {'BBox'}")
    print("-" * 88)

    for frame_path in frames:
        filename = os.path.basename(frame_path)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
            
        detected = detector.detect(frame)
        
        try:
            # We must first calibrate() to get the ice mask
            calibrated = calibrator.calibrate(frame)
            mask = calibrator.get_board_mask(frame)
            
            coverage, bbox = calculate_coverage_and_bbox(mask)
            
            # Save overlay
            # If mask is present, blend it with the frame
            overlay = frame.copy()
            if mask is not None:
                # Color the mask in green
                colored_mask = np.zeros_like(frame)
                colored_mask[mask > 0] = [0, 255, 0]
                overlay = cv2.addWeighted(frame, 0.7, colored_mask, 0.3, 0)
            
            out_path = f"tmp_frames/{filename.replace('.jpg', '_hybrid_gatefix.png')}"
            cv2.imwrite(out_path, overlay)
            
            print(f"{filename:<40} | {str(detected):<8} | {coverage*100:>5.1f}% | {bbox}")
        except Exception as e:
            print(f"{filename:<40} | {str(detected):<8} | ERROR  | {str(e)[:20]}")

if __name__ == '__main__':
    main()
