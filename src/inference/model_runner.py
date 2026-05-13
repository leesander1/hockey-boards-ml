import torch
import numpy as np
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.calibration.rink_calibrator import RinkCalibrator


class ModelRunner:
    def __init__(self, player_model_path="yolov8n-seg.pt", device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mock = False

        # ── Rink geometry calibrator (replaces OpenCV colour fallback) ───────
        self.calibrator = RinkCalibrator()
        self._calibrated = False

        # ── Player segmentation model ────────────────────────────────────────
        if YOLO is None:
            print("Ultralytics YOLO not installed. Using mock inference.")
            self.mock = True
            self.player_model = None
        else:
            print(f"Loading player segmentation model on {self.device}…")
            self.player_model = YOLO(player_model_path)

    # ── Board mask via rink-template homography ───────────────────────────────

    def get_board_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a binary mask of the rink boards using geometry-aware
        rink-template projection.

        Strategy
        --------
        • First frame: attempt full calibration from detected rink lines.
        • Subsequent frames: update homography via SIFT tracking.
        • If calibration ever fails, fall back to the OpenCV HSV heuristic.
        """
        if not self._calibrated:
            success = self.calibrator.calibrate(frame)
            if success:
                self._calibrated = True
                print("Rink calibration succeeded — using geometry-aware board mask.")
            else:
                print("Rink calibration failed — using HSV fallback for this frame.")
        else:
            self.calibrator.update_homography(frame)

        if self._calibrated:
            mask = self.calibrator.get_board_mask(frame)
            if mask is not None:
                return mask

        # ── HSV fallback (only used when calibration unavailable) ────────────
        return self._hsv_board_mask(frame)

    # ── Player mask via YOLOv8 segmentation ──────────────────────────────────

    def get_player_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a binary mask of all detected players (COCO class 0 = person).
        The mask is dilated slightly to avoid clipping player edges.
        """
        blank = np.zeros(frame.shape[:2], dtype=np.uint8)

        if self.mock or self.player_model is None:
            return blank

        results = self.player_model(frame, classes=[0], verbose=False)
        if not results or results[0].masks is None:
            return blank

        masks = results[0].masks.data.cpu().numpy()
        combined = np.any(masks, axis=0).astype(np.uint8) * 255
        combined = cv2.resize(combined, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Dilate to give players a small buffer (prevents edge clipping)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        combined = cv2.dilate(combined, kernel, iterations=1)
        return combined

    # ── HSV fallback ──────────────────────────────────────────────────────────

    def _hsv_board_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Detects the white ice surface and inverts it to get the boards.
        Constrains detection to the bottom 70% of the frame to exclude stands.
        """
        h, w = frame.shape[:2]
        roi_top = int(h * 0.15)   # ignore top 15% (stands / scoreboard)
        roi_bot = int(h * 0.85)   # ignore bottom 15% (camera housing / crowd)
        roi = frame[roi_top:roi_bot, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        ice_mask = cv2.inRange(hsv,
                               np.array([0,  0, 160]),
                               np.array([180, 40, 255]))

        k = np.ones((15, 15), np.uint8)
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_CLOSE, k)
        ice_mask = cv2.morphologyEx(ice_mask, cv2.MORPH_OPEN,  k)

        contours, _ = cv2.findContours(ice_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        board_mask = np.ones((h, w), dtype=np.uint8) * 255
        if contours:
            largest = max(contours, key=cv2.contourArea)
            largest[:, :, 1] += roi_top   # shift back to full-frame coords
            eps = 0.01 * cv2.arcLength(largest, True)
            approx = cv2.approxPolyDP(largest, eps, True)
            cv2.drawContours(board_mask, [approx], -1, 0, thickness=cv2.FILLED)

        # Zero out stands (top and bottom strips)
        board_mask[:roi_top, :] = 0
        board_mask[roi_bot:, :]  = 0
        return board_mask
