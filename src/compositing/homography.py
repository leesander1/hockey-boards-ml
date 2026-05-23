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

        # Soft blending weights are calculated dynamically at the end to support feathered edge blending

        # 1. Blur the original frame strongly to smear out sharp text/logo edges from the original ads.
        # This keeps the overall ambient lighting, gradients, and soft reflections without any readable text.
        k_size = int(w * 0.06) | 1
        blurred = cv2.GaussianBlur(frame, (k_size, k_size), 0)

        # 2. Desaturate the blurred image to neutralize the original ad colors,
        # while keeping the native color temperature/tint of the rink lighting (15% color, 85% gray).
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.merge([gray, gray, gray])
        ambient_light = cv2.addWeighted(blurred, 0.15, gray_3ch, 0.85, 0)

        # 3. Blend the crisp neutral texture with the processed ambient lighting layer.
        blended_layer = cv2.addWeighted(ad_layer, blend_alpha, ambient_light, 1.0 - blend_alpha, 0)

        # 4. Add Top Red Line and Bottom Yellow Line to the ad layer
        # Calculate thickness based on frame height (approx 0.3%, way smaller than 1.5%)
        thickness = max(1, int(h * 0.003))
        
        # Find top edge by shifting the mask down
        shifted_down = np.zeros_like(board_mask)
        shifted_down[thickness:, :] = board_mask[:-thickness, :]
        top_edge = cv2.bitwise_and(board_mask, cv2.bitwise_not(shifted_down))
        
        # Find bottom edge by shifting the mask up
        shifted_up = np.zeros_like(board_mask)
        shifted_up[:-thickness, :] = board_mask[thickness:, :]
        bottom_edge = cv2.bitwise_and(board_mask, cv2.bitwise_not(shifted_up))
        
        # Dynamically sample the red top and yellow bottom colors from the original frame (excluding players)
        top_edge_clean = cv2.bitwise_and(top_edge, cv2.bitwise_not(player_mask))
        bottom_edge_clean = cv2.bitwise_and(bottom_edge, cv2.bitwise_not(player_mask))
        
        top_pixels = frame[top_edge_clean > 0]
        if len(top_pixels) > 0:
            sampled_red = np.median(top_pixels, axis=0).astype(int)
            red_color = (int(sampled_red[0]), int(sampled_red[1]), int(sampled_red[2]))
        else:
            red_color = (30, 30, 200)      # Fallback: Darker red
            
        bottom_pixels = frame[bottom_edge_clean > 0]
        if len(bottom_pixels) > 0:
            sampled_yellow = np.median(bottom_pixels, axis=0).astype(int)
            yellow_color = (int(sampled_yellow[0]), int(sampled_yellow[1]), int(sampled_yellow[2]))
        else:
            yellow_color = (0, 200, 220)   # Fallback: Yellow
            
        # Create an overlay for the lines
        edge_overlay = np.zeros_like(blended_layer)
        edge_overlay[top_edge > 0] = red_color
        edge_overlay[bottom_edge > 0] = yellow_color
        
        # Create an edge mask for blending
        edge_mask = cv2.bitwise_or(top_edge, bottom_edge)
        # Blur the edge mask slightly to anti-alias and make it look smooth (avoid washing out thin lines)
        if thickness >= 5:
            edge_mask_blurred = cv2.GaussianBlur(edge_mask, (5, 5), 0)
        elif thickness >= 3:
            edge_mask_blurred = cv2.GaussianBlur(edge_mask, (3, 3), 0)
        else:
            edge_mask_blurred = edge_mask
        
        # Blend the lines into the blended_layer
        edge_mask_norm = edge_mask_blurred.astype(np.float32) / 255.0
        edge_mask_norm = np.expand_dims(edge_mask_norm, axis=-1)
        blended_layer = (edge_overlay * edge_mask_norm + blended_layer * (1.0 - edge_mask_norm)).astype(np.uint8)

        # Create a feathered (soft-edged) player mask to prevent hard clipping artifacts
        player_mask_blurred = cv2.GaussianBlur(player_mask, (3, 3), 0)
        P = player_mask_blurred.astype(np.float32) / 255.0
        
        # Slightly anti-alias the board mask edges too for seamless rink blending
        board_mask_blurred = cv2.GaussianBlur(board_mask, (3, 3), 0)
        B = board_mask_blurred.astype(np.float32) / 255.0
        
        # W represents the final soft-edged compositing weight (0.0 = player/original frame, 1.0 = replacement ad)
        W = B * (1.0 - P)
        W3 = np.expand_dims(W, axis=-1)  # shape (h, w, 1) for broadcasting
        
        # Blending equation: blended_layer * W3 + frame * (1.0 - W3)
        composited = blended_layer.astype(np.float32) * W3 + frame.astype(np.float32) * (1.0 - W3)
        return np.clip(composited, 0, 255).astype(np.uint8)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_ad_layer(self, h: int, w: int) -> np.ndarray:
        """Return the ad image tiled to (h, w) to preserve its resolution and detail, creating it once."""
        if self._base_ad is None or self._base_ad.shape[:2] != (h, w):
            if self.ad_image is not None:
                th, tw = self.ad_image.shape[:2]
                rep_y = int(np.ceil(h / th))
                rep_x = int(np.ceil(w / tw))
                tiled = np.tile(self.ad_image, (rep_y, rep_x, 1))
                # Make it "less white" (approx 80% brightness) to look natural under arena lighting
                self._base_ad = (tiled[:h, :w].astype(float) * 0.80).astype(np.uint8)
            else:
                # Green placeholder so the board region is clearly visible
                self._base_ad = np.zeros((h, w, 3), dtype=np.uint8)
                self._base_ad[:] = (0, 200, 0)
        return self._base_ad
