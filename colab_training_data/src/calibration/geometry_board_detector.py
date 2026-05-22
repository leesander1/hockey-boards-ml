#!/usr/bin/env python3
"""Geometry-based board detector using homography projection.

Projects the known board polygons from world coordinates to image coordinates,
avoiding unreliable color detection of the yellow trim.
"""

import cv2
import numpy as np
import json
import os

from src.calibration.rink_template import (
    get_near_board_polygon_world,
    get_far_board_polygon_world,
    get_end_board_polygons_world,
)


class GeometryBoardDetector:
    """Detects board regions by projecting known world geometry through a homography."""

    def __init__(self, homography_file: str | None = None):
        self._H = None  # homography matrix (3x3)
        self._board_mask = None

        if homography_file is None:
            homography_file = os.path.join(
                os.path.dirname(__file__), '..', 'src', 'calibration', 'annotation_homography.json'
            )

        if os.path.exists(homography_file):
            self._load_homography(homography_file)

    def _load_homography(self, path: str):
        """Load homography from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self._H = np.array(data['homography'], dtype=np.float32)
        except Exception as e:
            print(f'Warning: failed to load homography from {path}: {e}')

    def detect(self, frame: np.ndarray) -> bool:
        """Project board geometry onto frame. Returns True if homography is available."""
        if self._H is None:
            return False

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Project all board polygons
        board_polygons = [
            get_near_board_polygon_world(),
            get_far_board_polygon_world(),
        ]
        board_polygons.extend(get_end_board_polygons_world())

        for poly_world in board_polygons:
            pts_world = np.array(poly_world, dtype=np.float32).reshape(-1, 1, 2)
            # Perspective transform to image space
            pts_img = cv2.perspectiveTransform(pts_world, self._H)
            pts_img = pts_img.reshape(-1, 2).astype(np.int32)
            cv2.drawContours(mask, [pts_img], 0, 255, thickness=cv2.FILLED)

        # Spatial constraints
        mask[:int(h * 0.20), :int(w * 0.27)] = 0  # exclude scorebug

        # Morphological cleanup
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

        self._board_mask = mask
        return True

    def get_board_mask(self) -> np.ndarray | None:
        """Return the computed board mask."""
        return self._board_mask

    def is_ready(self) -> bool:
        """Returns True if homography is available."""
        return self._H is not None
