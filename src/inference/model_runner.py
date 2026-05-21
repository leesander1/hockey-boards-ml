import torch
import numpy as np
import cv2
import json

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from src.calibration.rink_calibrator import RinkCalibrator
from src.calibration.rink_anchor_fusion import RinkAnchorFusion


class ModelRunner:
    def __init__(
        self,
        player_model_path="yolov8n-seg.pt",
        device=None,
        hockeyai_model_path: str | None = None,
        hockeyrink_model_path: str | None = None,
        hockeyrink_keypoint_map_path: str | None = None,
        hockeyrink_keypoint_world_map: dict[int, tuple[float, float]] | None = None,
    ):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mock = False

        # ── Rink geometry calibrator (replaces OpenCV colour fallback) ───────
        self.calibrator = RinkCalibrator()
        self._calibrated = False
        self._anchor_fusion = RinkAnchorFusion()
        self._semantic_anchors = None
        self._keypoint_anchors = None
        self._feature_prior = None
        self._hockeyrink_keypoint_world_map = hockeyrink_keypoint_world_map or self._load_keypoint_world_map(hockeyrink_keypoint_map_path)
        self._hockeyai_model = None
        self._hockeyrink_model = None

        # ── Player segmentation model ────────────────────────────────────────
        if YOLO is None:
            print("Ultralytics YOLO not installed. Using mock inference.")
            self.mock = True
            self.player_model = None
        else:
            print(f"Loading player segmentation model on {self.device}…")
            self.player_model = YOLO(player_model_path)

        if YOLO is None:
            if hockeyai_model_path is not None or hockeyrink_model_path is not None:
                print("Ultralytics YOLO not installed. HockeyAI/HockeyRink models disabled.")
        else:
            if hockeyai_model_path:
                print(f"Loading HockeyAI detector on {self.device}…")
                self._hockeyai_model = YOLO(hockeyai_model_path)
            if hockeyrink_model_path:
                print(f"Loading HockeyRink pose model on {self.device}…")
                self._hockeyrink_model = YOLO(hockeyrink_model_path)

    @staticmethod
    def _load_keypoint_world_map(path: str | None) -> dict[int, tuple[float, float]]:
        """Load a keypoint-to-world map from JSON if provided."""
        if not path:
            return {}

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Warning: failed to load HockeyRink keypoint map from {path}: {exc}")
            return {}

        mapping: dict[int, tuple[float, float]] = {}
        if isinstance(data, dict):
            for key, value in data.items():
                try:
                    idx = int(key)
                    if isinstance(value, (list, tuple)) and len(value) == 2:
                        mapping[idx] = (float(value[0]), float(value[1]))
                except Exception:
                    continue

        if not mapping:
            print(f"Warning: HockeyRink keypoint map at {path} did not contain usable index mappings.")
        return mapping

    def set_semantic_anchors(self, detections):
        """Provide HockeyAI-style semantic detections for board alignment."""
        self._semantic_anchors = detections
        self._keypoint_anchors = None

    def set_hockeyai_results(self, results):
        """Provide raw HockeyAI detector results and normalize them automatically."""
        detections = self._anchor_fusion.hockeyai_detections_from_results(results)
        self.set_semantic_anchors(detections)

    def set_rink_keypoints(self, keypoints, keypoint_world_map, confidences=None):
        """Provide HockeyRink keypoints for board alignment."""
        self._keypoint_anchors = {
            'keypoints': keypoints,
            'keypoint_world_map': keypoint_world_map,
            'confidences': confidences,
        }
        self._semantic_anchors = None

    def set_hockeyrink_results(self, results, keypoint_world_map):
        """Provide raw HockeyRink pose results and normalize them automatically."""
        keypoints, confidences = self._anchor_fusion.hockeyrink_keypoints_from_results(results)
        self.set_rink_keypoints(keypoints, keypoint_world_map, confidences=confidences)

    def set_hockeyrink_keypoint_world_map(self, keypoint_world_map: dict[int, tuple[float, float]]):
        """Set the keypoint-to-world mapping used by HockeyRink pose anchors."""
        self._hockeyrink_keypoint_world_map = keypoint_world_map

    def clear_rink_anchors(self):
        """Clear any externally supplied rink anchors and priors."""
        self._semantic_anchors = None
        self._keypoint_anchors = None
        self._feature_prior = None
        self.calibrator.set_feature_prior(None)

    def _update_rink_anchors_from_models(self, frame: np.ndarray):
        """Run optional HockeyAI/HockeyRink models and update the feature prior."""
        if self.mock:
            return

        if self._hockeyai_model is not None:
            ai_results = self._hockeyai_model(frame, verbose=False)
            detections = self._anchor_fusion.hockeyai_detections_from_results(ai_results)
            if detections:
                self._semantic_anchors = detections
                self._keypoint_anchors = None
        # HockeyRink pose model integration has been disabled.
        # If you want to re-enable pose-based anchors, set them manually
        # via `set_rink_keypoints()` or restore this block.

    def _update_feature_prior(self, frame_shape: tuple[int, int]):
        """Build and cache a board prior from any supplied rink anchors."""
        prior = None

        if self._semantic_anchors is not None:
            prior = self._anchor_fusion.build_prior_from_semantics(self._semantic_anchors, frame_shape)
        elif self._keypoint_anchors is not None:
            prior = self._anchor_fusion.build_prior_from_keypoints(
                self._keypoint_anchors.get('keypoints'),
                self._keypoint_anchors.get('keypoint_world_map', {}),
                frame_shape,
                confidences=self._keypoint_anchors.get('confidences'),
            )

        self._feature_prior = prior
        self.calibrator.set_feature_prior(prior)

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
        self._update_rink_anchors_from_models(frame)
        self._update_feature_prior(frame.shape[:2])

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
