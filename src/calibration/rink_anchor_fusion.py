#!/usr/bin/env python3
"""Fuse rink semantic features and keypoints into a board prior.

The HockeyAI dataset contributes semantic anchors like centerIce, faceoff, and
goal detections. The HockeyRink model contributes rink keypoints that can be
mapped to world coordinates once a stable keypoint order is known.

This helper keeps the math separate from the board detector: it estimates a
homography from any supplied image/world correspondences and converts that into
a soft board prior mask using the canonical rink template geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from src.calibration.rink_template import (
    CALIBRATION_WORLD_POINTS,
    BLUE_LINE_LEFT_X,
    BLUE_LINE_RIGHT_X,
    CENTER_LINE_X,
    RINK_WIDTH,
    get_near_board_polygon_world,
    get_far_board_polygon_world,
    get_end_board_polygons_world,
)


@dataclass(frozen=True)
class AnchorObservation:
    """A matched image/world point pair."""

    label: str
    image_point: tuple[float, float]
    world_point: tuple[float, float]
    confidence: float = 1.0


def _as_box_center(item) -> tuple[float, float] | None:
    if item is None:
        return None

    if isinstance(item, dict):
        if 'point' in item:
            x, y = item['point']
            return float(x), float(y)
        if 'bbox' in item:
            x1, y1, x2, y2 = item['bbox']
            return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)
        if all(k in item for k in ('x', 'y')):
            return float(item['x']), float(item['y'])

    if isinstance(item, (tuple, list)):
        if len(item) == 2:
            return float(item[0]), float(item[1])
        if len(item) == 4:
            x1, y1, x2, y2 = item
            return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    return None


def _first_item(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) > 0:
        return value[0]
    return value


class RinkAnchorFusion:
    """Estimate a homography and board prior from rink anchors."""

    def __init__(self):
        self._last_homography: np.ndarray | None = None
        self._last_prior: np.ndarray | None = None

    @staticmethod
    def default_world_points() -> dict[str, tuple[float, float]]:
        """Return a conservative anchor-to-world map for common rink features."""
        return {
            'centerIce': (CENTER_LINE_X, RINK_WIDTH / 2.0),
            # Approximate faceoff-circle centers provide non-collinear anchors.
            'faceoff_top_left': (69.0, 22.5),
            'faceoff_bottom_left': (69.0, 62.5),
            'faceoff_top_right': (131.0, 22.5),
            'faceoff_bottom_right': (131.0, 62.5),
            'goal_left': (11.0, 22.5),
            'goal_right': (189.0, 62.5),
            'blue_left_near': (BLUE_LINE_LEFT_X, 0.0),
            'blue_left_far': (BLUE_LINE_LEFT_X, RINK_WIDTH),
            'blue_right_near': (BLUE_LINE_RIGHT_X, 0.0),
            'blue_right_far': (BLUE_LINE_RIGHT_X, RINK_WIDTH),
        }

    def observations_from_semantics(self, detections, frame_shape: tuple[int, int]) -> list[AnchorObservation]:
        """Convert HockeyAI-style semantic detections into anchor observations.

        Expected input shape is intentionally loose:
        - dict label -> list of boxes/points
        - list of dicts with keys like label/bbox/point/confidence
        """
        h, w = frame_shape
        world_map = self.default_world_points()
        observations: list[AnchorObservation] = []

        def add_obs(label: str, image_point, confidence: float = 1.0):
            if label not in world_map:
                return
            pt = _as_box_center(image_point)
            if pt is None:
                return
            observations.append(
                AnchorObservation(
                    label=label,
                    image_point=pt,
                    world_point=world_map[label],
                    confidence=float(confidence),
                )
            )

        if isinstance(detections, dict):
            for label, values in detections.items():
                if label in ('goal', 'faceoff') and isinstance(values, list):
                    for idx, value in enumerate(values):
                        # Split ambiguous labels by approximate image quadrant.
                        pt = _as_box_center(value)
                        if pt is None:
                            continue
                        x, y = pt
                        if label == 'goal':
                            side_label = 'goal_left' if x < (w / 2.0) else 'goal_right'
                        else:
                            if x < (w / 2.0) and y < (h / 2.0):
                                side_label = 'faceoff_top_left'
                            elif x < (w / 2.0) and y >= (h / 2.0):
                                side_label = 'faceoff_bottom_left'
                            elif x >= (w / 2.0) and y < (h / 2.0):
                                side_label = 'faceoff_top_right'
                            else:
                                side_label = 'faceoff_bottom_right'
                        if side_label in world_map:
                            observations.append(
                                AnchorObservation(
                                    label=side_label,
                                    image_point=pt,
                                    world_point=world_map[side_label],
                                    confidence=1.0,
                                )
                            )
                else:
                    add_obs(label, values, 1.0)
        else:
            for item in detections or []:
                if not isinstance(item, dict):
                    continue
                label = item.get('label') or item.get('class')
                if label is None:
                    continue
                point = item.get('point') or item.get('bbox') or item
                confidence = float(item.get('confidence', 1.0))
                add_obs(label, point, confidence)

        # Heuristic fallback: if we only have one goal/faceoff detection on each
        # side, assign them using x-position so the homography can still be fit.
        observations.sort(key=lambda obs: obs.image_point[0])
        if len(observations) >= 2:
            # Re-label left/right ambiguous detections against the image center.
            center_x = w / 2.0
            remapped: list[AnchorObservation] = []
            for obs in observations:
                if obs.label in ('goal', 'faceoff'):
                    side = 'left' if obs.image_point[0] < center_x else 'right'
                    key = f'{obs.label}_{side}'
                    if key in world_map:
                        remapped.append(
                            AnchorObservation(
                                label=key,
                                image_point=obs.image_point,
                                world_point=world_map[key],
                                confidence=obs.confidence,
                            )
                        )
                        continue
                remapped.append(obs)
            observations = remapped

        return observations

    def hockeyai_detections_from_results(self, results) -> dict[str, list[dict[str, object]]] | None:
        """Normalize Ultralytics-style detection results into semantic detections.

        This is intentionally lightweight and only depends on duck-typed fields
        commonly exposed by Ultralytics Results objects.
        """
        result = _first_item(results)
        if result is None:
            return None

        boxes = getattr(result, 'boxes', None)
        if boxes is None:
            return None

        xyxy = getattr(boxes, 'xyxy', None)
        conf = getattr(boxes, 'conf', None)
        cls = getattr(boxes, 'cls', None)
        if xyxy is None or cls is None:
            return None

        xyxy = xyxy.detach().cpu().numpy() if hasattr(xyxy, 'detach') else np.asarray(xyxy)
        cls = cls.detach().cpu().numpy() if hasattr(cls, 'detach') else np.asarray(cls)
        conf = conf.detach().cpu().numpy() if conf is not None and hasattr(conf, 'detach') else conf
        names = getattr(result, 'names', None) or getattr(getattr(result, 'model', None), 'names', None) or {}

        detections: dict[str, list[dict[str, object]]] = {}
        for idx, box in enumerate(xyxy):
            class_id = int(cls[idx])
            label = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
            confidence = float(conf[idx]) if conf is not None and len(conf) > idx else 1.0
            detections.setdefault(label, []).append(
                {
                    'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    'confidence': confidence,
                }
            )

        return detections

    def hockeyrink_keypoints_from_results(self, results) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Extract keypoint arrays from Ultralytics-style pose results."""
        result = _first_item(results)
        if result is None:
            return None, None

        keypoints = getattr(result, 'keypoints', None)
        if keypoints is None:
            return None, None

        xy = getattr(keypoints, 'xy', None)
        conf = getattr(keypoints, 'conf', None)
        if xy is None:
            return None, None

        xy = xy.detach().cpu().numpy() if hasattr(xy, 'detach') else np.asarray(xy)
        xy = np.asarray(xy, dtype=np.float32)

        conf_arr = None
        if conf is not None:
            conf_arr = conf.detach().cpu().numpy() if hasattr(conf, 'detach') else np.asarray(conf)
            conf_arr = np.asarray(conf_arr, dtype=np.float32)

        # Ultralytics pose outputs typically return shape (1, K, 2) for one image.
        if xy.ndim == 3 and xy.shape[0] == 1:
            xy = xy[0]
        if conf_arr is not None and conf_arr.ndim == 3 and conf_arr.shape[0] == 1:
            conf_arr = conf_arr[0]

        return xy, conf_arr

    def observations_from_keypoints(
        self,
        keypoints,
        keypoint_world_map: dict[int, tuple[float, float]],
        confidences: Iterable[float] | None = None,
    ) -> list[AnchorObservation]:
        """Convert a keypoint array into world/image correspondences.

        The caller must provide `keypoint_world_map` because the public HockeyRink
        card does not publish the 56-keypoint ordering.
        """
        if keypoints is None:
            return []

        pts = np.asarray(keypoints, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return []

        # Normalize confidences into a 1D list of floats (one per keypoint).
        if confidences is None:
            confs = [1.0] * len(pts)
        else:
            arr = np.asarray(confidences)
            # If confidences come as shape (N,1) or (1,N,1) etc., try to collapse
            if arr.ndim == 1:
                confs = [float(x) for x in arr.tolist()]
            elif arr.ndim >= 2:
                try:
                    arr2 = arr.reshape(len(pts), -1)
                    confs = [float(row[0]) for row in arr2]
                except Exception:
                    flat = arr.flatten()
                    confs = [float(flat[i]) if i < len(flat) else 1.0 for i in range(len(pts))]
            else:
                confs = [1.0] * len(pts)
        observations: list[AnchorObservation] = []
        for idx, world_point in keypoint_world_map.items():
            if idx < 0 or idx >= len(pts):
                continue
            x, y = pts[idx, :2]
            if np.isnan(x) or np.isnan(y):
                continue
            conf = float(confs[idx]) if idx < len(confs) else 1.0
            observations.append(
                AnchorObservation(
                    label=f'keypoint_{idx}',
                    image_point=(float(x), float(y)),
                    world_point=(float(world_point[0]), float(world_point[1])),
                    confidence=conf,
                )
            )
        return observations

    def estimate_homography(self, observations: list[AnchorObservation]) -> np.ndarray | None:
        """Estimate a homography from anchor observations."""
        if len(observations) < 4:
            return None

        world_pts = np.array([obs.world_point for obs in observations], dtype=np.float32)
        image_pts = np.array([obs.image_point for obs in observations], dtype=np.float32)
        weights = np.array([max(0.1, obs.confidence) for obs in observations], dtype=np.float32)

        # Weighted RANSAC isn't directly exposed, but we can scale the effective
        # sample support by keeping only the most reliable observations.
        order = np.argsort(weights)[::-1]
        keep = order[: max(4, min(len(order), 12))]
        H, _ = cv2.findHomography(world_pts[keep], image_pts[keep], cv2.RANSAC, 4.0)
        self._last_homography = H
        return H

    def build_board_prior_from_homography(self, H: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray | None:
        """Warp the canonical board polygons into image space to create a soft prior."""
        if H is None:
            return None

        h, w = frame_shape
        prior = np.zeros((h, w), dtype=np.uint8)

        board_polygons = [
            get_near_board_polygon_world(),
            get_far_board_polygon_world(),
        ]
        board_polygons.extend(get_end_board_polygons_world())

        for poly_world in board_polygons:
            pts_world = np.array(poly_world, dtype=np.float32).reshape(-1, 1, 2)
            pts_img = cv2.perspectiveTransform(pts_world, H).reshape(-1, 2)
            pts_img = np.round(pts_img).astype(np.int32)
            cv2.fillConvexPoly(prior, pts_img, 255)

        prior = cv2.GaussianBlur(prior, (0, 0), 3.0)
        self._last_prior = prior.astype(np.float32) / 255.0
        return self._last_prior

    def build_prior_from_semantics(self, detections, frame_shape: tuple[int, int]) -> np.ndarray | None:
        observations = self.observations_from_semantics(detections, frame_shape)
        H = self.estimate_homography(observations)
        if H is None:
            return None
        return self.build_board_prior_from_homography(H, frame_shape)

    def build_prior_from_keypoints(
        self,
        keypoints,
        keypoint_world_map: dict[int, tuple[float, float]],
        frame_shape: tuple[int, int],
        confidences: Iterable[float] | None = None,
    ) -> np.ndarray | None:
        observations = self.observations_from_keypoints(keypoints, keypoint_world_map, confidences=confidences)
        H = self.estimate_homography(observations)
        if H is None:
            return None
        return self.build_board_prior_from_homography(H, frame_shape)

    def last_homography(self) -> np.ndarray | None:
        return self._last_homography

    def last_prior(self) -> np.ndarray | None:
        return self._last_prior