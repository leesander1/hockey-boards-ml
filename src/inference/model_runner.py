import torch
import numpy as np
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ModelRunner:
    def __init__(self, board_model_path="yolov8n-seg.pt", player_model_path="yolov8n-seg.pt", device=None):
        if YOLO is None:
            print("Ultralytics YOLO is not installed. Using mock inference.")
            self.mock = True
            return
            
        self.mock = False
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading models on {self.device}...")
        # In a real scenario, these would be two different models or a single multi-class model
        # Currently using the same pre-trained model for demonstration
        self.board_model = YOLO(board_model_path)
        self.player_model = YOLO(player_model_path)
        
    def get_board_mask(self, frame):
        """
        Runs the board segmentation model and returns a binary mask of the rink boards.
        """
        if self.mock:
            # Return empty mask (all zeros) for demonstration
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
        results = self.board_model(frame, classes=[0], verbose=False) # Example: class 0
        if len(results) > 0 and results[0].masks is not None:
            # Combine all masks into one binary mask
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            # Resize mask to match original frame shape if necessary
            import cv2
            combined_mask = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            return combined_mask
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
    def get_player_mask(self, frame):
        """
        Runs the player instance segmentation model and returns a binary mask of all players/objects.
        """
        if self.mock:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            
        # Class 0 usually maps to 'person' in COCO dataset
        results = self.player_model(frame, classes=[0], verbose=False) 
        if len(results) > 0 and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            import cv2
            combined_mask = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
            return combined_mask
        return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
