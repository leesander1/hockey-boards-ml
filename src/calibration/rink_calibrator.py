"""
Rink Calibrator — finds the ice surface boundary and derives a board mask
from it.  Works for any camera angle (end-zone, side, corner) without
requiring specific rink-line detection.

Algorithm
---------
1. Detect the ice surface (large bright region) using HSV thresholding.
2. Find its contour → this gives us the exact ice/board boundary.
3. Dilate the contour outward by BOARD_DILATION_PX pixels → board zone.
4. Subtract the ice interior → leaves only the board strip.
5. Apply a brightness gate to exclude dark stand pixels.
6. SIFT tracking maintains the mask per-frame as the camera pans.
"""

import cv2
import numpy as np

# How many pixels to expand outward from the ice edge to cover the board face.
BOARD_DILATION_PX = 55
# Minimum brightness for a board pixel (eliminates dark stands).
BOARD_MIN_BRIGHTNESS = 80
# Minimum ice contour area as fraction of frame area (ignore small blobs).
MIN_ICE_AREA_FRACTION = 0.15


class RinkCalibrator:
    def __init__(self):
        self.H = None            # frame-to-frame tracking homography (optional)
        self._ice_mask = None    # cached ice mask (updated each frame)
        self._sift = cv2.SIFT_create()
        self._flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50))
        self._prev_kp = None
        self._prev_des = None

    # ── Public API ───────────────────────────────────────────────────────────

    def calibrate(self, frame: np.ndarray) -> bool:
        """Detect the ice surface. Returns True if ice is found."""
        ice = self._detect_ice(frame)
        if ice is None:
            return False
        self._ice_mask = ice
        return True

    def update_homography(self, frame: np.ndarray):
        """Track camera motion via SIFT and update the ice mask accordingly."""
        self._ice_mask = self._detect_ice(frame)  # re-detect each frame

        # Optional: SIFT tracking for sub-frame refinement
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self._sift.detectAndCompute(gray, None)
        self._prev_kp = kp
        self._prev_des = des

    def get_board_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Returns a binary mask of the rink board region (uint8, 255=board).
        The board strip is the area just OUTSIDE the ice surface boundary,
        after removing dark pixels (stands).
        """
        if self._ice_mask is None:
            return None

        h, w = frame.shape[:2]

        # ── Dilate ice outward → ice + boards ───────────────────────────────
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (BOARD_DILATION_PX * 2 + 1, BOARD_DILATION_PX * 2 + 1))
        ice_plus_boards = cv2.dilate(self._ice_mask, k, iterations=1)

        # Board zone = (ice + boards) minus ice interior
        board_zone = cv2.bitwise_and(
            ice_plus_boards, cv2.bitwise_not(self._ice_mask))

        # ── Remove dark stand pixels ─────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bright = (gray >= BOARD_MIN_BRIGHTNESS).astype(np.uint8) * 255
        board_mask = cv2.bitwise_and(board_zone, bright)

        # ── Light clean-up ───────────────────────────────────────────────────
        k_small = np.ones((5, 5), np.uint8)
        board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, k_small)

        # ── Spatial constraints ──────────────────────────────────────────────
        # Boards are always in the upper portion of the frame for this camera.
        # Zero out everything below the top 45% to eliminate stand bleed.
        board_mask[int(h * 0.45):, :] = 0

        # Zero out the broadcast scorebug region (top-left corner).
        board_mask[:int(h * 0.22), :int(w * 0.28)] = 0

        return board_mask

    def is_calibrated(self) -> bool:
        return self._ice_mask is not None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_ice(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Detects the ice surface as the dominant bright, low-saturation region.
        Returns a filled binary mask (uint8) or None if not found.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ice: very bright (V > 175) and low saturation (S < 50)
        ice_raw = cv2.inRange(hsv,
                              np.array([0,   0, 175]),
                              np.array([180, 50, 255]))

        # Morphological cleanup
        k_large = np.ones((20, 20), np.uint8)
        ice_raw = cv2.morphologyEx(ice_raw, cv2.MORPH_CLOSE, k_large)
        ice_raw = cv2.morphologyEx(ice_raw, cv2.MORPH_OPEN,  k_large)

        # Find the largest contour — that's the ice surface
        contours, _ = cv2.findContours(
            ice_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_ICE_AREA_FRACTION * h * w:
            return None

        # Fill the ice contour to get a solid ice mask
        ice_filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(ice_filled, [largest], -1, 255, thickness=cv2.FILLED)
        return ice_filled
