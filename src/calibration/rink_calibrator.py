"""
Rink Calibrator — Computes a homography from rink world coords (feet) to
image pixel coords by auto-detecting the red center line and two blue lines.

In a standard broadcast side-camera view the rink lines appear as roughly
*vertical* stripes in the image (they run across the 85-ft width, which is the
depth axis of the frame).

Detection pipeline
------------------
1. HSV-filter for red  → one  vertical stripe  (center line,  x=100 ft)
2. HSV-filter for blue → two  vertical stripes (blue  lines, x=75 / 125 ft)
3. For each detected stripe, record:
     • x_img_top    = x pixel where the stripe meets the far  boards (top of ice)
     • x_img_bottom = x pixel where the stripe meets the near boards (bottom of ice)
     • y_img_top    = y pixel of that intersection
     • y_img_bottom = y pixel of that intersection
4. 6 image/world point pairs → findHomography (RANSAC)
"""

import cv2
import numpy as np
from .rink_template import (
    BLUE_LINE_LEFT_X, BLUE_LINE_RIGHT_X, CENTER_LINE_X,
    RINK_WIDTH, RINK_LENGTH,
    get_near_board_polygon_world, get_far_board_polygon_world,
)


class RinkCalibrator:
    def __init__(self):
        self.H = None          # homography: world (ft) → image (px)
        self._last_gray = None
        self._sift = cv2.SIFT_create()
        self._flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5), dict(checks=50)
        )
        self._prev_kp = None
        self._prev_des = None

    # ── Public API ───────────────────────────────────────────────────────────

    def calibrate(self, frame: np.ndarray) -> bool:
        """
        Try to calibrate from a single frame by detecting rink lines.
        Returns True if a valid homography was computed.
        """
        red_mask  = self._color_mask(frame, color="red")
        blue_mask = self._color_mask(frame, color="blue")

        red_stripe  = self._detect_vertical_stripe(frame, red_mask)
        blue_stripes = self._detect_two_vertical_stripes(frame, blue_mask)

        if red_stripe is None or blue_stripes is None:
            return False

        blue_left, blue_right = blue_stripes

        # Build correspondence: (world_x, world_y) ↔ (img_x, img_y)
        # world_y=0 is near boards (appears at BOTTOM of ice → larger y pixel)
        # world_y=RINK_WIDTH is far boards (appears at TOP → smaller y pixel)
        world_pts = np.float32([
            [BLUE_LINE_LEFT_X,   0.0],
            [BLUE_LINE_LEFT_X,   RINK_WIDTH],
            [CENTER_LINE_X,      0.0],
            [CENTER_LINE_X,      RINK_WIDTH],
            [BLUE_LINE_RIGHT_X,  0.0],
            [BLUE_LINE_RIGHT_X,  RINK_WIDTH],
        ])

        img_pts = np.float32([
            blue_left["bottom"],
            blue_left["top"],
            red_stripe["bottom"],
            red_stripe["top"],
            blue_right["bottom"],
            blue_right["top"],
        ])

        H, mask = cv2.findHomography(world_pts, img_pts, cv2.RANSAC, 5.0)
        if H is None or mask is None or mask.sum() < 4:
            return False

        self.H = H
        return True

    def update_homography(self, frame: np.ndarray):
        """
        After initial calibration, update H by tracking SIFT features frame
        to frame (handles camera pans without re-running full calibration).
        """
        if self.H is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self._sift.detectAndCompute(gray, None)

        if (self._prev_des is not None and des is not None
                and len(self._prev_kp) > 10 and len(kp) > 10):
            matches = self._flann.knnMatch(self._prev_des, des, k=2)
            good = [m for m_n in matches if len(m_n) == 2
                    for m, n in [m_n] if m.distance < 0.8 * n.distance]

            if len(good) >= 10:
                src = np.float32([self._prev_kp[m.queryIdx].pt
                                  for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp[m.trainIdx].pt
                                  for m in good]).reshape(-1, 1, 2)
                M, _ = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                if M is not None:
                    # Update: new_H maps world → current frame
                    self.H = M @ self.H

        self._prev_kp  = kp
        self._prev_des = des

    def get_board_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Projects the board polygons (defined in world coords) onto the image
        using the calibrated homography.  Returns a binary uint8 mask.
        """
        if self.H is None:
            return None

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for poly_world in [get_near_board_polygon_world(),
                           get_far_board_polygon_world()]:
            pts_w = np.float32(poly_world).reshape(-1, 1, 2)
            pts_i = cv2.perspectiveTransform(pts_w, self.H)
            pts_i = np.int32(pts_i).reshape(-1, 2)
            cv2.fillPoly(mask, [pts_i], 255)

        # Clip to frame bounds
        mask[:, :] = np.clip(mask, 0, 255)
        return mask

    def is_calibrated(self) -> bool:
        return self.H is not None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _color_mask(self, frame: np.ndarray, color: str) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if color == "red":
            m1 = cv2.inRange(hsv, np.array([0,  120, 100]),
                                  np.array([10, 255, 255]))
            m2 = cv2.inRange(hsv, np.array([168, 120, 100]),
                                  np.array([180, 255, 255]))
            mask = cv2.bitwise_or(m1, m2)
        elif color == "blue":
            mask = cv2.inRange(hsv, np.array([90, 80,  60]),
                                    np.array([130, 255, 255]))
        else:
            raise ValueError(f"Unknown color: {color}")

        # Clean up noise
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    def _detect_vertical_stripe(self, frame: np.ndarray,
                                 mask: np.ndarray) -> dict | None:
        """
        Finds the dominant vertical stripe in a binary mask.
        Returns {"top": (x,y), "bottom": (x,y)} in image coords.
        """
        h, w = mask.shape[:2]

        # Project mask horizontally → find the dominant x column
        col_sum = mask.sum(axis=0).astype(np.float32)
        if col_sum.max() == 0:
            return None

        # Smooth and find peak
        col_sum = cv2.GaussianBlur(col_sum.reshape(1, -1),
                                   (1, 31), 0).flatten()
        stripe_x = int(np.argmax(col_sum))

        # Find the topmost and bottommost white pixel in that column ±10px band
        band = mask[:, max(0, stripe_x-10): min(w, stripe_x+10)]
        rows = np.where(band.any(axis=1))[0]
        if len(rows) < 10:
            return None

        top_y    = int(rows[0])
        bottom_y = int(rows[-1])

        return {
            "top":    (stripe_x, top_y),
            "bottom": (stripe_x, bottom_y),
            "x":      stripe_x,
        }

    def _detect_two_vertical_stripes(self, frame: np.ndarray,
                                      mask: np.ndarray):
        """
        Finds the two dominant vertical stripes in a binary mask
        (the left and right blue lines).
        Returns (left_stripe, right_stripe) or None.
        """
        h, w = mask.shape[:2]
        col_sum = mask.sum(axis=0).astype(np.float32)
        if col_sum.max() == 0:
            return None

        col_smooth = cv2.GaussianBlur(col_sum.reshape(1, -1),
                                      (1, 31), 0).flatten()

        # Find two peaks separated by at least 20% of frame width
        peaks = []
        tmp = col_smooth.copy()
        for _ in range(2):
            pk = int(np.argmax(tmp))
            if tmp[pk] == 0:
                break
            peaks.append(pk)
            # suppress neighbourhood
            lo = max(0,   pk - w // 5)
            hi = min(w-1, pk + w // 5)
            tmp[lo:hi] = 0

        if len(peaks) < 2:
            return None

        peaks.sort()
        results = []
        for px in peaks:
            band = mask[:, max(0, px-10): min(w, px+10)]
            rows = np.where(band.any(axis=1))[0]
            if len(rows) < 10:
                return None
            results.append({
                "top":    (px, int(rows[0])),
                "bottom": (px, int(rows[-1])),
                "x":      px,
            })

        return results[0], results[1]   # left stripe, right stripe
