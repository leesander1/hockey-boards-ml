import cv2
import numpy as np

class AdCompositor:
    def __init__(self, ad_image_path=None):
        self.ad_image = None
        if ad_image_path:
            self.ad_image = cv2.imread(ad_image_path)
            if self.ad_image is None:
                print(f"Warning: Could not load ad image from {ad_image_path}")

    def apply_ad(self, frame, board_mask, player_mask):
        """
        Replaces the board area with the ad, while keeping the players on top.
        """
        # If no ad is provided, just blank out the boards using a solid color (e.g., green)
        if self.ad_image is None:
            # Create a blank green ad
            ad_layer = np.zeros_like(frame)
            ad_layer[:] = (0, 255, 0) # BGR
        else:
            # For a proper homography transform, we would detect keypoints to warp the ad_image.
            # As a simplified placeholder, we resize the ad to fit the frame
            ad_layer = cv2.resize(self.ad_image, (frame.shape[1], frame.shape[0]))
        
        # In a full implementation:
        # H, _ = cv2.findHomography(src_pts, dst_pts)
        # ad_layer = cv2.warpPerspective(self.ad_image, H, (frame.shape[1], frame.shape[0]))

        # The region where we WANT to place the ad is the board_mask
        # But we MUST NOT place it where the player_mask is active
        
        # Calculate the final insertion mask: board_mask AND NOT player_mask
        insertion_mask = cv2.bitwise_and(board_mask, cv2.bitwise_not(player_mask))
        
        # Create a 3-channel version of the insertion mask
        insertion_mask_3c = cv2.merge([insertion_mask, insertion_mask, insertion_mask])
        
        # Blend the ad_layer into the frame using the insertion mask
        # Where insertion_mask is 255, we take ad_layer. Where it's 0, we take original frame.
        foreground = cv2.bitwise_and(ad_layer, insertion_mask_3c)
        background = cv2.bitwise_and(frame, cv2.bitwise_not(insertion_mask_3c))
        
        composited_frame = cv2.add(foreground, background)
        
        return composited_frame
