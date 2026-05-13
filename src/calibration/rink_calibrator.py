"""
Rink Calibrator — finds the ice surface boundary and derives a board mask.

Algorithm
---------
1. Detect ice: HSV threshold for bright, low-saturation pixels.
2. Convex-hull the ice contour → smooth, convex ice boundary (no jagged edges).
3. Geometry-aware board polygon:
   a. Extract the TOP edge of the ice hull (far-boards side).
   b. For each point on that edge, offset it UPWARD by a perspective-scaled
      amount that approximates the physical 42-inch (3.5 ft) board height.
   c. Fill the resulting quadrilateral polygon → board mask zone.
4. Color-refine: within the polygon, keep board-like pixels.
5. Morphological cleanup for smooth final edges.
"""

import cv2
import numpy as np
from scipy.ndimage import median_filter

# ── Tuning constants ──────────────────────────────────────────────────────────
# Board height in pixels at top (far boards) and bottom (near boards) of frame.
# Physical boards are 42 inches = 3.5 ft tall. Perspective foreshortening means
# they appear SHORTER near the top of the frame and TALLER near the bottom.
BOARD_HEIGHT_TOP_PX    = 22    # pixels of board face visible at top of frame
BOARD_HEIGHT_BOTTOM_PX = 90    # pixels of board face visible at bottom of frame

BOARD_MIN_V         = 65     # min HSV Value (brightness) — not dark stands
BOARD_MAX_S         = 255    # max saturation — allow yellow kickplates etc.
MIN_ICE_AREA_FRAC   = 0.15   # smallest fraction of frame that counts as ice
ICE_MORPH_K         = 20     # kernel size for ice morphological clean-up


class RinkCalibrator:
    def __init__(self):
        self._ice_hull = None    # convex hull of ice contour (pts, shape=(N,1,2))
        self._ice_mask = None    # filled ice mask (uint8)

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(self, frame: np.ndarray) -> bool:
        """Detect ice and build convex hull. Returns True if ice is found."""
        result = self._detect_ice(frame)
        if result is None:
            return False
        self._ice_mask, self._ice_hull = result
        return True

    def update_homography(self, frame: np.ndarray):
        """Re-detect ice every frame (handles camera pans automatically)."""
        result = self._detect_ice(frame)
        if result is not None:
            self._ice_mask, self._ice_hull = result

    def get_board_mask(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Returns a refined binary board mask (uint8, 255 = board ad zone).

        Steps
        -----
        1. Extract the top-edge points from the ice convex hull.
        2. Build a board polygon: top-edge shifted upward by perspective-scaled
           board height, bottom = the ice top edge itself.
        3. Colour gate: only board-like pixels within the polygon survive.
        4. Morphological cleanup.
        """
        if self._ice_mask is None:
            return None

        h, w = frame.shape[:2]

        # ── 1. Find the topmost ice pixel per column ──────────────────────────
        # We use the filled ice mask column-by-column for the main profile, but
        # also rasterize the CONVEX HULL top edge as a fallback floor — so that
        # crease/net gaps in the raw ice mask don't cause the strip to spike
        # upward into the stands.
        ice = self._ice_mask  # (h, w) uint8
        has_ice = ice > 0     # (h, w) bool

        col_has_ice = has_ice.any(axis=0)                       # (w,)
        top_ice_y = np.where(
            col_has_ice,
            np.argmax(has_ice, axis=0),
            h
        ).astype(np.float32)                                    # (w,)

        # Also draw the filled convex hull as a reference and rasterize its top edge.
        # The hull is always convex and won't have crease gaps.
        hull_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(hull_mask, [self._ice_hull], -1, 255, thickness=cv2.FILLED)
        has_hull = hull_mask > 0
        col_has_hull = has_hull.any(axis=0)
        hull_top_y = np.where(
            col_has_hull,
            np.argmax(has_hull, axis=0),
            h
        ).astype(np.float32)

        # Use the LOWER boundary (larger y = further down) of the two profiles
        # so crease/net gaps can't push the strip above where the boards really are.
        top_ice_y = np.maximum(top_ice_y, hull_top_y)

        # Smooth the top-ice profile to remove small pixel noise
        smooth_win = max(7, w // 15) | 1
        top_ice_y = median_filter(top_ice_y, size=smooth_win).astype(np.float32)

        # ── 2. Build board zone column-by-column ──────────────────────────────
        # For each column x, the boards occupy rows [top_ice_y[x] - board_h, top_ice_y[x]].
        # The board height in pixels scales linearly with the row position.
        board_zone = np.zeros((h, w), dtype=np.uint8)

        # Vectorised: compute per-column board_h and top-of-boards row
        t_col = top_ice_y / h    # 0 = frame top, 1 = frame bottom
        board_h_col = (BOARD_HEIGHT_TOP_PX
                       + t_col * (BOARD_HEIGHT_BOTTOM_PX - BOARD_HEIGHT_TOP_PX)
                      ).astype(np.float32)

        board_top_y = np.clip(top_ice_y - board_h_col, 0, h - 1).astype(np.float32)
        board_bot_y = np.clip(top_ice_y, 0, h - 1).astype(np.float32)

        # Smooth the top boundary once more to eliminate remaining spikes
        # Use a wide window — this is the critical line that controls the glass edge
        top_smooth_win = max(11, w // 10) | 1
        board_top_y = median_filter(board_top_y, size=top_smooth_win).astype(np.float32)
        board_bot_y = median_filter(board_bot_y, size=top_smooth_win // 2 | 1).astype(np.float32)

        # Absolute floor: the board strip can never start above 8% of the frame height
        board_top_y = np.clip(board_top_y, h * 0.08, h - 1).astype(np.int32)
        board_bot_y = board_bot_y.astype(np.int32)

        # Fill the board zone mask column by column (vectorised)
        row_idx = np.arange(h, dtype=np.int32)[:, None]   # (h, 1)
        in_board = (row_idx >= board_top_y[None, :]) & (row_idx <= board_bot_y[None, :])
        # Only include columns that actually have ice
        in_board &= col_has_ice[None, :]
        board_zone[in_board] = 255

        # ── 3. Colour gate ────────────────────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        s_channel = hsv[:, :, 1]

        board_color_gate = (
            (v_channel >= BOARD_MIN_V) &          # not dark stands
            (s_channel <= BOARD_MAX_S)            # not hyper-saturated noise
        ).astype(np.uint8) * 255

        board_mask = cv2.bitwise_and(board_zone, board_color_gate)

        # ── 4. Spatial constraints ─────────────────────────────────────────────
        # Exclude broadcast scorebug (top-left corner)
        board_mask[:int(h * 0.20), :int(w * 0.27)] = 0

        # ── 5. Morphological cleanup ──────────────────────────────────────────
        # Close small gaps inside the board strip (bridges lettering cutouts)
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 9))
        board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_CLOSE, k_close)
        # Open to remove thin noisy filaments
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        board_mask = cv2.morphologyEx(board_mask, cv2.MORPH_OPEN, k_open)
        # Light erode to tighten edges
        k_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        board_mask = cv2.erode(board_mask, k_erode, iterations=1)

        # ── 6. Column-wise top-edge smoothing ─────────────────────────────────
        # Even with a polygon, jagged glass reflections can create notches at the
        # top. Run a 1D median filter along the topmost board row per column.
        has_board = (board_mask > 0)
        first_row = np.where(
            has_board.any(axis=0),
            np.argmax(has_board, axis=0),
            h
        ).astype(np.int32)

        win = max(3, w // 12) | 1
        top_smooth = median_filter(first_row.astype(float), size=win).astype(np.int32)

        row_idx = np.arange(h, dtype=np.int32)[:, None]
        above   = row_idx < top_smooth[None, :]
        board_mask[above] = 0

        return board_mask

    def is_calibrated(self) -> bool:
        return self._ice_mask is not None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _detect_ice(self, frame: np.ndarray):
        """
        Detects the ice as the largest bright low-saturation region.
        Returns (ice_mask_filled, convex_hull_pts) or None.
        """
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ice: very bright (V > 175) and low saturation (S < 50)
        ice_raw = cv2.inRange(hsv,
                              np.array([0,   0, 175]),
                              np.array([180, 50, 255]))

        # Morphological clean-up to join nearby regions and fill small holes
        k = np.ones((ICE_MORPH_K, ICE_MORPH_K), np.uint8)
        ice_raw = cv2.morphologyEx(ice_raw, cv2.MORPH_CLOSE, k)
        ice_raw = cv2.morphologyEx(ice_raw, cv2.MORPH_OPEN,  k)

        contours, _ = cv2.findContours(
            ice_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < MIN_ICE_AREA_FRAC * h * w:
            return None

        # ── Convex hull → smooth boundary ─────────────────────────────────
        hull = cv2.convexHull(largest)

        ice_filled = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(ice_filled, [hull], -1, 255, thickness=cv2.FILLED)

        return ice_filled, hull
