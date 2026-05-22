#!/usr/bin/env python3
"""Example-based board detector using feature-matched mask transfer.

This detector matches the current frame to annotated template frames and warps
the template board mask into the current view using a homography.
"""

import os
from glob import glob

import cv2
import numpy as np


class ExampleBasedBoardDetector:
    """Transfer masks from annotated templates using ORB + homography."""

    def __init__(self, annotation_dir: str | None = None):
        if annotation_dir is None:
            annotation_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'annotation_frames')

        self._templates = []
        self._board_mask = None

        self._orb = cv2.ORB_create(nfeatures=2000, fastThreshold=12)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._load_templates(annotation_dir)

    def _load_templates(self, annotation_dir: str):
        """Load template JPG frames with matching annotated PNG masks."""
        jpg_paths = sorted(glob(os.path.join(annotation_dir, '*.jpg')))

        for jpg_path in jpg_paths:
            stem = os.path.splitext(os.path.basename(jpg_path))[0]
            png_path = os.path.join(annotation_dir, f'{stem}.png')
            if not os.path.exists(png_path):
                continue

            frame = cv2.imread(jpg_path)
            ann = cv2.imread(png_path)
            if frame is None or ann is None:
                continue

            mask = self._build_mask_from_annotation(ann)
            if mask is None or (mask > 0).mean() < 0.01:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kps, desc = self._orb.detectAndCompute(gray, None)
            if desc is None or len(kps) < 20:
                continue

            self._templates.append(
                {
                    'name': stem,
                    'shape': frame.shape[:2],
                    'keypoints': kps,
                    'descriptors': desc,
                    'mask': mask,
                }
            )

    def detect(self, frame: np.ndarray) -> bool:
        """Detect board mask by matching to templates and warping template mask."""
        self._board_mask = None

        if not self._templates:
            return False

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        q_kps, q_desc = self._orb.detectAndCompute(gray, None)
        if q_desc is None or len(q_kps) < 20:
            return False

        best = None
        best_score = -1.0

        for tmpl in self._templates:
            matches = self._bf.knnMatch(tmpl['descriptors'], q_desc, k=2)
            good = []
            for pair in matches:
                if len(pair) != 2:
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

            if len(good) < 20:
                continue

            src = np.float32([tmpl['keypoints'][m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([q_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            H, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if H is None or inlier_mask is None:
                continue

            inliers = int(inlier_mask.sum())
            inlier_ratio = inliers / max(len(good), 1)
            score = inliers * inlier_ratio

            if score > best_score:
                best_score = score
                best = (tmpl, H, inliers, inlier_ratio)

        if best is None:
            return False

        tmpl, H, inliers, inlier_ratio = best

        # Strong acceptance gate: only trust high-quality geometric matches.
        if inliers < 45 or inlier_ratio < 0.50:
            return False

        warped = cv2.warpPerspective(
            tmpl['mask'],
            H,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Cleanup and scorebug exclusion.
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
        warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, k_close)
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        warped = cv2.morphologyEx(warped, cv2.MORPH_OPEN, k_open)
        warped[: int(h * 0.20), : int(w * 0.27)] = 0

        cov = float((warped > 0).mean())
        if cov < 0.02 or cov > 0.12:
            return False

        ys, xs = np.where(warped > 0)
        if xs.size == 0:
            return False

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        height = y_max - y_min + 1

        # Reject implausible board bands (full-frame spill or too-tall warps).
        if y_min < int(h * 0.05):
            return False
        if height > int(h * 0.55):
            return False
        if x_min == 0 and x_max == w - 1 and y_min < int(h * 0.12) and y_max > int(h * 0.70):
            return False

        self._board_mask = warped
        return True

    def get_board_mask(self) -> np.ndarray | None:
        """Return detected board mask."""
        return self._board_mask

    def template_count(self) -> int:
        """Return number of loaded templates."""
        return len(self._templates)

    @staticmethod
    def _build_mask_from_annotation(ann_bgr: np.ndarray) -> np.ndarray | None:
        """Build a dense board mask from red/yellow annotation lines."""
        hsv = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2HSV)
        h, w = ann_bgr.shape[:2]

        red = cv2.inRange(hsv, np.array([0, 150, 150], np.uint8), np.array([10, 255, 255], np.uint8))
        red = cv2.bitwise_or(red, cv2.inRange(hsv, np.array([170, 150, 150], np.uint8), np.array([180, 255, 255], np.uint8)))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

        yellow = cv2.inRange(hsv, np.array([18, 150, 150], np.uint8), np.array([35, 255, 255], np.uint8))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

        top = np.full(w, np.nan, dtype=np.float32)
        bot = np.full(w, np.nan, dtype=np.float32)
        for col in range(w):
            r_ys = np.where(red[:, col] > 0)[0]
            y_ys = np.where(yellow[:, col] > 0)[0]
            if r_ys.size:
                top[col] = float(r_ys.min())
            if y_ys.size:
                bot[col] = float(y_ys.max())

        valid = ~(np.isnan(top) | np.isnan(bot))
        if valid.sum() < w * 0.1:
            return None

        xs = np.arange(w)
        if np.isnan(top).any():
            top[np.isnan(top)] = np.interp(xs[np.isnan(top)], xs[~np.isnan(top)], top[~np.isnan(top)])
        if np.isnan(bot).any():
            bot[np.isnan(bot)] = np.interp(xs[np.isnan(bot)], xs[~np.isnan(bot)], bot[~np.isnan(bot)])

        top = np.clip(top, 0, h - 1).astype(np.int32)
        bot = np.clip(bot, 0, h - 1).astype(np.int32)

        row = np.arange(h, dtype=np.int32)[:, None]
        mask = ((row >= top[None, :]) & (row <= bot[None, :])).astype(np.uint8) * 255
        mask[: int(h * 0.25), : int(w * 0.35)] = 0
        return mask