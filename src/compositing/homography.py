"""
AdCompositor — blends a replacement ad into the board mask area.

Camera tracking (SIFT / homography) is now owned by RinkCalibrator.
This class just receives a ready-made board_mask and player_mask per frame
and does the pixel-level compositing.
"""

import cv2
import numpy as np


class AdCompositor:
    def __init__(self, ad_image_path: str | None = None):
        self.ad_image = None
        if ad_image_path:
            self.ad_image = cv2.imread(ad_image_path)
            if self.ad_image is None:
                print(f"Warning: could not load ad image from {ad_image_path}. "
                      "Using green fallback.")
        self._base_ad = None   # lazily initialised to frame size

    # ── Main compositing entry point ──────────────────────────────────────────

    def apply_ad(self,
                 frame: np.ndarray,
                 board_mask: np.ndarray,
                 player_mask: np.ndarray,
                 blend_alpha: float = 0.70) -> np.ndarray:
        """
        Composite the replacement ad onto `frame` inside `board_mask`,
        but *behind* any players indicated by `player_mask`.

        Parameters
        ----------
        frame       : BGR image from the video stream
        board_mask  : uint8 binary mask — 255 where boards are
        player_mask : uint8 binary mask — 255 where players are
        blend_alpha : float — blending opacity for the new ad/board texture (0.0 to 1.0)

        Returns
        -------
        Composited BGR frame.
        """
        h, w = frame.shape[:2]
        ad_layer = self._get_ad_layer(h, w)

        # Final insertion region: boards that are NOT occluded by players
        insertion_mask = cv2.bitwise_and(
            board_mask, cv2.bitwise_not(player_mask)
        )
        _, insertion_mask = cv2.threshold(
            insertion_mask, 127, 255, cv2.THRESH_BINARY
        )

        mask3 = cv2.merge([insertion_mask] * 3)

        # Blend the ad/neutral texture with the original frame's lighting and background
        blended_layer = cv2.addWeighted(ad_layer, blend_alpha, frame, 1.0 - blend_alpha, 0)

        foreground = cv2.bitwise_and(blended_layer, mask3)
        background = cv2.bitwise_and(frame,    cv2.bitwise_not(mask3))
        return cv2.add(foreground, background)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_ad_layer(self, h: int, w: int) -> np.ndarray:
        """Return the ad image resized to (h, w), creating it once."""
        if self._base_ad is None or self._base_ad.shape[:2] != (h, w):
            if self.ad_image is not None:
                self._base_ad = cv2.resize(self.ad_image, (w, h))
            else:
                # Green placeholder so the board region is clearly visible
                self._base_ad = np.zeros((h, w, 3), dtype=np.uint8)
                self._base_ad[:] = (0, 200, 0)
        return self._base_ad
