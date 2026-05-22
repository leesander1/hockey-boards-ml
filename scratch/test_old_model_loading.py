import cv2
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.calibration.ml_board_detector import MLBoardDetector

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
    print("Success! Model loaded successfully.")

if __name__ == '__main__':
    main()
