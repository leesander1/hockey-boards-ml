#!/usr/bin/env python3
"""Learn a soft board-shape prior from annotated mask frames.

The annotation folder already contains representative board masks. Averaging
them yields a spatial prior that can bias the ML output toward the known board
ribbon shape without requiring a new model.
"""

from __future__ import annotations

import os
from glob import glob

import cv2
import numpy as np


TARGET_WIDTH = 320
TARGET_HEIGHT = 176


class BoardSpatialPrior:
    """Loads a soft prior mask for the board region from annotated masks."""

    def __init__(self, annotation_dir: str | None = None):
        if annotation_dir is None:
            annotation_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'annotation_frames', 'new_batch', 'annotated'
            )

        self._prior_map: np.ndarray | None = None
        self._annotation_dir = annotation_dir
        self._load_prior()

    def _load_prior(self):
        mask_paths = sorted(glob(os.path.join(self._annotation_dir, '*-mask.png')))
        if not mask_paths:
            return

        accum = np.zeros((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.float32)
        count = 0

        for path in mask_paths:
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            mask = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
            accum += (mask > 0).astype(np.float32)
            count += 1

        if count == 0:
            return

        prior = accum / float(count)
        prior = cv2.GaussianBlur(prior, (0, 0), 2.5)
        self._prior_map = np.clip(prior, 0.0, 1.0)

    def is_ready(self) -> bool:
        return self._prior_map is not None

    def get_prior_map(self, shape: tuple[int, int]) -> np.ndarray | None:
        if self._prior_map is None:
            return None

        h, w = shape
        if self._prior_map.shape == (h, w):
            return self._prior_map

        prior = cv2.resize(self._prior_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(prior, 0.0, 1.0)