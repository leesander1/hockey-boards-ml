"""
Board Trim Detector — finds the dasher board region using the distinctive
yellow/gold kickplate trim that runs along the ice–board junction in most
NHL rinks.

Strategy
--------
1. Detect the "trim" color band:  a yellow/gold hue (HSV H≈15-45) that appears
   at the base of the boards, just above the ice surface.
2. For each column, find the vertical centroid of that trim band → gives us
   `trim_y[x]`, the y-coordinate of the board base across the full width.
3. Smooth `trim_y` with a wide median filter to get a stable board-bottom profile.
4. Project UPWARD from `trim_y` by a perspective-scaled board height to get
   `board_top_y[x]`.
5. Fill the strip [board_top_y, trim_y] per column → board zone mask.
6. Fallback: if trim detection fails (no trim visible in this shot), fall back to
   the ice-boundary method from RinkCalibrator.

Tuning constants at top of file — adjust per arena if the trim is a different hue.
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter
import json
import os

# ── Trim colour (yellow/gold kickplate) ───────────────────────────────────────
# Calibrated from manual annotations across 6 reference frames.
# Two arena trim styles observed:
#   - Yellow/gold kickplate (H≈15-45) — most common
#   - Orange-red dasher bottom trim (H≈0-15, 165-180)
# We use a wider hue range and rely on the ice-mask spatial constraint to avoid
# false positives from the crowd or jerseys.
TRIM_H_LO = 10    # Hue lower bound (OpenCV: 0-180)
TRIM_H_HI = 50    # Hue upper bound — covers yellow through orange-gold
TRIM_S_LO = 80    # Saturation — must be reasonably saturated
TRIM_S_HI = 255
TRIM_V_LO = 100   # Brightness — not too dark
TRIM_V_HI = 255

# Separate range for orange-red trim (VGK arena, etc.)
TRIM_RED_LO1 = np.array([0,  100, 100])
TRIM_RED_HI1 = np.array([10, 255, 255])
TRIM_RED_LO2 = np.array([168, 100, 100])
TRIM_RED_HI2 = np.array([180, 255, 255])

# Minimum number of trim pixels per column for the column to be "valid"
TRIM_MIN_PX_PER_COL = 3

# Board height in pixels at top vs bottom of frame (perspective foreshortening).
# Physical boards are 42 inches tall. Near the top of the frame (far ice) they
# appear shorter; near the bottom (near ice) they appear taller.
# Values calibrated from manual annotations of 6 broadcast frames.
BOARD_HEIGHT_TOP_PX    = 15   # far boards (y ~ 0)  — calibrated from annotations
BOARD_HEIGHT_BOTTOM_PX = 190  # near boards (y ~ h) — calibrated from annotations

# Fraction of frame height above which we never place the board top edge
BOARD_MAX_TOP_FRAC = 0.08   # 8% — scorebug / stands ceiling


class TrimDetector:
    """
    Detects the dasher board strip using the yellow/gold trim at the board base.
    Can be used standalone or as a drop-in module inside RinkCalibrator.
    """

    def __init__(self):
        self._trim_mask: np.ndarray | None = None
        self._trim_y: np.ndarray | None = None   # per-column board-bottom y
        self._board_top_y: np.ndarray | None = None
        self._board_bot_y: np.ndarray | None = None
        self._calibration_cache: tuple[float, float] | None = None

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, ice_mask: np.ndarray | None = None) -> bool:
        """
        Run trim detection on `frame`.

        Parameters
        ----------
        frame     : BGR image
        ice_mask  : optional filled ice mask (uint8).  When provided, trim pixels
                    outside the ice region +/- a small margin are discarded, which
                    helps exclude stand reflections that share the yellow hue.

        Returns True if a usable trim band was found.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── 1. Colour gate (yellow/gold + orange-red trim) ────────────────────
        yel_mask = cv2.inRange(
            hsv,
            np.array([TRIM_H_LO, TRIM_S_LO, TRIM_V_LO], dtype=np.uint8),
            np.array([TRIM_H_HI, TRIM_S_HI, TRIM_V_HI], dtype=np.uint8),
        )
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, TRIM_RED_LO1, TRIM_RED_HI1),
            cv2.inRange(hsv, TRIM_RED_LO2, TRIM_RED_HI2),
        )
        trim_raw = cv2.bitwise_or(yel_mask, red_mask)

        # ── 2. Spatial mask: restrict to a band near the ice boundary ─────────
        if ice_mask is not None:
            # Dilate ice mask downward (into boards) and upward (a little into
            # the boards above ice) to create a valid search zone for trim.
            kd = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 80))
            ku = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
            ice_down  = cv2.dilate(ice_mask, kd)     # extend downward
            ice_up    = cv2.dilate(ice_mask, ku)      # extend upward
            search_zone = cv2.bitwise_or(ice_down, ice_up)
            trim_raw = cv2.bitwise_and(trim_raw, search_zone)

        # Also restrict to the top 70% of the frame — trim is never in the crowd
        trim_raw[int(h * 0.70):, :] = 0
        # Exclude scorebug zone
        trim_raw[:int(h * 0.20), :int(w * 0.27)] = 0

        self._trim_mask = trim_raw

        # ── 3. Per-column centroid → trim_y profile ────────────────────────────
        has_trim = trim_raw > 0
        col_trim_count = has_trim.sum(axis=0).astype(np.float32)   # (w,)
        valid_cols = col_trim_count >= TRIM_MIN_PX_PER_COL

        # Weighted centroid y per column
        row_weights = np.arange(h, dtype=np.float32)[:, None]      # (h,1)
        col_sum_y   = (has_trim * row_weights).sum(axis=0)          # (w,)

        # Centroid (weighted mean row of trim pixels)
        trim_y_raw = np.where(valid_cols, col_sum_y / np.where(col_trim_count > 0, col_trim_count, 1), np.nan)

        valid_frac = valid_cols.mean()
        if valid_frac < 0.10:
            # Less than 10% of columns have any trim — detection failed
            return False

        # Interpolate NaN gaps (columns with no trim) linearly
        trim_y_raw = self._interp_nans(trim_y_raw, h)

        # Wide median filter for stability
        smooth_win = max(11, w // 12) | 1
        trim_y_sm = median_filter(trim_y_raw, size=smooth_win).astype(np.float32)

        self._trim_y = trim_y_sm   # board BOTTOM per column

        # ── 4. Project upward → board top ─────────────────────────────────────
        t_col   = trim_y_sm / h   # 0 = top of frame, 1 = bottom

        top_px, bottom_px = self._get_board_height_calibration()

        board_h = (top_px + t_col * (bottom_px - top_px))

        raw_top = trim_y_sm - board_h

        # Second median pass on the top edge for extra smoothness
        top_win = max(15, w // 8) | 1
        board_top = median_filter(raw_top, size=top_win).astype(np.float32)

        # Clamp to frame
        board_top = np.clip(board_top, h * BOARD_MAX_TOP_FRAC, h - 1).astype(np.int32)
        board_bot = np.clip(trim_y_sm, 0, h - 1).astype(np.int32)

        self._board_top_y = board_top
        self._board_bot_y = board_bot
        return True

    def get_board_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Returns the binary board mask (uint8, 255 = board ad zone).
        Must call detect() first.
        """
        if self._board_top_y is None:
            return None

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        row_idx = np.arange(h, dtype=np.int32)[:, None]
        in_board = (row_idx >= self._board_top_y[None, :]) & (row_idx <= self._board_bot_y[None, :])
        mask[in_board] = 255

        # Morphological cleanup
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        return mask

    def get_trim_y(self) -> np.ndarray | None:
        """Per-column board-bottom y-profile (the detected trim centroid)."""
        return self._trim_y

    def get_trim_mask(self) -> np.ndarray | None:
        """Raw binary trim colour mask (for debugging)."""
        return self._trim_mask

    def is_detected(self) -> bool:
        return self._trim_y is not None

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _interp_nans(arr: np.ndarray, fill_val: float) -> np.ndarray:
        """Linear interpolation of NaN gaps; fills edges with nearest valid value."""
        arr = arr.copy()
        nan_mask = np.isnan(arr)
        if not nan_mask.any():
            return arr
        if nan_mask.all():
            arr[:] = fill_val
            return arr
        xs = np.arange(len(arr))
        valid_x = xs[~nan_mask]
        valid_v = arr[~nan_mask]
        arr[nan_mask] = np.interp(xs[nan_mask], valid_x, valid_v)
        return arr

    def _get_board_height_calibration(self) -> tuple[float, float]:
        """Load annotation-based board height calibration once, with sanity bounds."""
        if self._calibration_cache is not None:
            return self._calibration_cache

        top_px = BOARD_HEIGHT_TOP_PX
        bottom_px = BOARD_HEIGHT_BOTTOM_PX

        ann_file = os.path.join(os.path.dirname(__file__), 'annotation_calibration.json')
        if os.path.exists(ann_file):
            try:
                with open(ann_file, 'r') as f:
                    ann = json.load(f)
                top_px = float(ann.get('board_height_top_px', top_px))
                bottom_px = float(ann.get('board_height_bottom_px', bottom_px))
            except Exception:
                top_px = BOARD_HEIGHT_TOP_PX
                bottom_px = BOARD_HEIGHT_BOTTOM_PX

        top_px = float(np.clip(top_px, 20.0, 120.0))
        bottom_px = float(np.clip(bottom_px, top_px + 10.0, 160.0))
        self._calibration_cache = (top_px, bottom_px)
        return self._calibration_cache
