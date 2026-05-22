#!/usr/bin/env python3
"""Auxiliary rink-feature detector for blue lines and faceoff circles.

This detector is intentionally lightweight: it looks for the strong blue
markings and a few large circular structures that often puncture the white ice
mask. The output is used as support to keep the ice hull from collapsing around
those markings.
"""

from __future__ import annotations

import cv2
import numpy as np


class RinkFeatureDetector:
    """Detects rink markings that are useful as auxiliary ice support."""

    def __init__(self):
        self._feature_mask: np.ndarray | None = None
        self._confidence_score = 0.0

    def detect(self, frame: np.ndarray) -> bool:
        """Detect blue lines and large circles. Returns True on any support."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Blue rink markings are usually strong and moderately saturated.
        blue = cv2.inRange(
            hsv,
            np.array([90, 70, 40], dtype=np.uint8),
            np.array([135, 255, 255], dtype=np.uint8),
        )

        # Ignore the broadcast scorebug and extreme frame edges.
        blue[:int(h * 0.12), :int(w * 0.35)] = 0
        blue[:int(h * 0.08), :] = 0
        blue[int(h * 0.92):, :] = 0

        blue = cv2.morphologyEx(
            blue,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
        )
        blue = cv2.morphologyEx(
            blue,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
        )

        support = np.zeros((h, w), dtype=np.uint8)

        # Blue lines show up as roughly vertical bands in the broadcast view.
        edges = cv2.Canny(blue, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=60,
            minLineLength=max(40, int(h * 0.18)),
            maxLineGap=18,
        )
        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, line)
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                length = float(np.hypot(dx, dy))
                # Keep the near-vertical, rink-like strokes only.
                if dy >= dx * 1.8 and length >= max(60.0, 0.20 * h):
                    cv2.line(support, (x1, y1), (x2, y2), 255, thickness=11)

        # Faceoff circles/creases provide additional support when visible.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 1.8)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(40, w // 6),
            param1=120,
            param2=28,
            minRadius=max(18, int(min(h, w) * 0.04)),
            maxRadius=max(36, int(min(h, w) * 0.20)),
        )
        if circles is not None:
            circles = np.round(circles[0]).astype(int)
            for x, y, r in circles:
                if 0 <= x < w and 0 <= y < h and r > 0:
                    cv2.circle(support, (x, y), int(r * 1.05), 255, thickness=8)

        support = cv2.morphologyEx(
            support,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )

        self._feature_mask = support
        self._confidence_score = float((support > 0).mean())
        return self._confidence_score > 0.002

    def get_feature_mask(self) -> np.ndarray | None:
        """Return the latest auxiliary support mask."""
        return self._feature_mask

    def get_confidence_score(self) -> float:
        """Return a coarse support confidence score."""
        return self._confidence_score

    def is_ready(self) -> bool:
        """Returns True when the detector has produced a mask."""
        return self._feature_mask is not None