#!/usr/bin/env python3
"""Annotation-based board detector using direct line extraction.

Instead of computing a homography, we extract the red (top) and yellow (bottom)
annotation lines and use them as the direct board boundary for that frame.

This is perfect for generating ground truth for model training or for frames
that have been annotated.
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter


class AnnotationBoardDetector:
    """Detects boards using manually-drawn red (top) and yellow (bottom) lines."""

    def __init__(self):
        self._board_mask = None
        self._board_top_y = None
        self._board_bot_y = None

    def detect(self, frame: np.ndarray) -> bool:
        """Extract red and yellow annotation lines and build board mask."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Only run annotation extraction when marker-like colors are present.
        # This avoids false positives on normal broadcast frames.
        if not self._has_annotation_signature(hsv):
            return False

        # Red annotation (board top)
        red = cv2.inRange(hsv, np.array([0, 150, 150], np.uint8), np.array([10, 255, 255], np.uint8))
        red = cv2.bitwise_or(red, cv2.inRange(hsv, np.array([170, 150, 150], np.uint8), np.array([180, 255, 255], np.uint8)))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

        # Yellow annotation (board bottom)
        yellow = cv2.inRange(hsv, np.array([18, 150, 150], np.uint8), np.array([35, 255, 255], np.uint8))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

        # Extract per-column extremes
        red_top_y = np.full(w, np.nan, dtype=np.float32)
        yellow_bot_y = np.full(w, np.nan, dtype=np.float32)

        for col in range(w):
            r_ys = np.where(red[:, col] > 0)[0]
            y_ys = np.where(yellow[:, col] > 0)[0]
            if r_ys.size:
                red_top_y[col] = float(r_ys.min())
            if y_ys.size:
                yellow_bot_y[col] = float(y_ys.max())

        # Find the column range where lines actually exist (to cut off at rink ends)
        valid_cols = np.where(~(np.isnan(red_top_y) | np.isnan(yellow_bot_y)))[0]
        if len(valid_cols) < w * 0.1:  # less than 10% valid
            return False
        
        # Mark columns outside the annotation range as NaN to exclude board-end regions
        ann_col_lo = valid_cols.min()
        ann_col_hi = valid_cols.max()
        red_top_y[:ann_col_lo] = np.nan
        red_top_y[ann_col_hi + 1:] = np.nan
        yellow_bot_y[:ann_col_lo] = np.nan
        yellow_bot_y[ann_col_hi + 1:] = np.nan

        # Interpolate NaNs
        xs = np.arange(w)
        red_top_y = self._interp_nans(red_top_y, h)
        yellow_bot_y = self._interp_nans(yellow_bot_y, h)

        # Smooth
        smooth_win = max(11, w // 64) | 1
        red_top_y = median_filter(red_top_y, size=smooth_win).astype(np.float32)
        yellow_bot_y = median_filter(yellow_bot_y, size=smooth_win).astype(np.float32)

        # Clamp to frame
        red_top_y = np.clip(red_top_y, 0, h - 1).astype(np.int32)
        yellow_bot_y = np.clip(yellow_bot_y, 0, h - 1).astype(np.int32)

        self._board_top_y = red_top_y
        self._board_bot_y = yellow_bot_y

        # Build mask
        mask = np.zeros((h, w), dtype=np.uint8)
        row_idx = np.arange(h, dtype=np.int32)[:, None]
        in_board = (row_idx >= red_top_y[None, :]) & (row_idx <= yellow_bot_y[None, :])
        mask[in_board] = 255

        # Morphological cleanup
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        # Exclude scorebug zone (top-left, more aggressive)
        mask[:int(h * 0.25), :int(w * 0.35)] = 0
        
        # Also exclude the very top of the frame (stands above glass) — use the red line as ceiling
        for col in range(w):
            if not np.isnan(red_top_y[col]):
                top_safe = max(0, int(red_top_y[col]) - 5)  # small margin above red line
                if top_safe < h * 0.12:  # only if in upper part of frame
                    mask[:top_safe, col] = 0

        self._board_mask = mask
        return True

    @staticmethod
    def _has_annotation_signature(hsv: np.ndarray) -> bool:
        """Return True when frame likely contains manual annotation marker strokes."""
        h, w = hsv.shape[:2]

        # Strict marker-like color gate: very saturated, bright pen strokes.
        red_strict = cv2.inRange(hsv, np.array([0, 220, 180], np.uint8), np.array([8, 255, 255], np.uint8))
        red_strict = cv2.bitwise_or(
            red_strict,
            cv2.inRange(hsv, np.array([172, 220, 180], np.uint8), np.array([180, 255, 255], np.uint8)),
        )
        yellow_strict = cv2.inRange(hsv, np.array([20, 220, 180], np.uint8), np.array([33, 255, 255], np.uint8))

        red_cols = (red_strict > 0).any(axis=0)
        yellow_cols = (yellow_strict > 0).any(axis=0)
        valid_cols = red_cols & yellow_cols

        red_frac = float((red_strict > 0).mean())
        yellow_frac = float((yellow_strict > 0).mean())
        valid_frac = float(valid_cols.mean())

        return red_frac >= 0.0005 and yellow_frac >= 0.0005 and valid_frac >= 0.03

    def get_board_mask(self) -> np.ndarray | None:
        """Return the computed board mask."""
        return self._board_mask

    def get_board_top_y(self) -> np.ndarray | None:
        """Return per-column board top y-coordinates."""
        return self._board_top_y

    def get_board_bot_y(self) -> np.ndarray | None:
        """Return per-column board bottom y-coordinates."""
        return self._board_bot_y

    @staticmethod
    def _interp_nans(arr: np.ndarray, fill_val: float) -> np.ndarray:
        """Linear interpolation of NaN gaps."""
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
