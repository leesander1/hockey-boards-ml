#!/usr/bin/env python3
"""Board detector using a trained segmentation model."""

import cv2
import numpy as np
import torch
import torch.nn as nn
import os
import json
import torch.nn.functional as F
from scipy.ndimage import median_filter
import torchvision.models.segmentation as segmentation

from src.calibration.board_spatial_prior import BoardSpatialPrior

TARGET_WIDTH = 640
TARGET_HEIGHT = 360
BOARD_HEIGHT_TOP_PX = 15
BOARD_HEIGHT_BOTTOM_PX = 190
BOARD_MAX_TOP_FRAC = 0.08


class UNet(nn.Module):
    """Lightweight U-Net for board segmentation."""

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = self._conv_block(128, 256)

        # Decoder
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(64, 32)

        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # Bottleneck
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        up3 = self.upconv3(b)
        if up3.shape != e3.shape:
            up3 = F.interpolate(up3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))

        up2 = self.upconv2(d3)
        if up2.shape != e2.shape:
            up2 = F.interpolate(up2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = self.upconv1(d2)
        if up1.shape != e1.shape:
            up1 = F.interpolate(up1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        return self.sigmoid(self.final(d1))


def get_deeplabv3_model(pretrained=True, num_classes=1, aux_loss=True):
    """Constructs a DeepLabV3 model with a MobileNetV3-Large backbone adapted for binary mask."""
    if pretrained:
        try:
            from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights
            weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
            model = segmentation.deeplabv3_mobilenet_v3_large(weights=weights, aux_loss=aux_loss)
        except Exception:
            model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, aux_loss=aux_loss)
    else:
        model = segmentation.deeplabv3_mobilenet_v3_large(weights=None, aux_loss=aux_loss)
        
    # Replace classifier head
    in_channels = model.classifier[4].in_channels
    model.classifier[4] = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    # Replace aux classifier head if present
    if aux_loss and model.aux_classifier is not None:
        in_channels_aux = model.aux_classifier[4].in_channels
        model.aux_classifier[4] = nn.Conv2d(in_channels_aux, num_classes, kernel_size=1)
        
    return model


class MLBoardDetector:
    """Board detector using trained segmentation model."""

    def __init__(self, model_path: str | None = None):
        self._model = None
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._board_mask = None
        self._prob_map = None
        self._confidence_score = 0.0
        self._spatial_prior = BoardSpatialPrior()
        self._feature_prior = None

        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'board_segmentation_model.pth')

        if os.path.exists(model_path):
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load trained model from disk."""
        try:
            # Determine architecture from config if available
            config_path = path.replace('.pth', '_config.json')
            architecture = 'unet'
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        cfg = json.load(f)
                    architecture = cfg.get('architecture', 'unet')
                except Exception as e:
                    print(f'Warning: failed to read config {config_path}: {e}')

            if architecture == 'deeplabv3':
                # Attempt to load with aux_loss=True first (default from notebook)
                try:
                    self._model = get_deeplabv3_model(pretrained=False, num_classes=1, aux_loss=True).to(self._device)
                    self._model.load_state_dict(torch.load(path, map_location=self._device))
                except Exception as e:
                    # If that fails (e.g. if saved without auxiliary classifier), try aux_loss=False
                    self._model = get_deeplabv3_model(pretrained=False, num_classes=1, aux_loss=False).to(self._device)
                    self._model.load_state_dict(torch.load(path, map_location=self._device))
            else:
                self._model = UNet(in_channels=3, out_channels=1).to(self._device)
                self._model.load_state_dict(torch.load(path, map_location=self._device))

            self._model.eval()
            print(f'Loaded {architecture} board segmentation model from {path}')
        except Exception as e:
            print(f'Warning: failed to load model from {path}: {e}')

    def set_feature_prior(self, prior_mask: np.ndarray | None):
        """Set an externally computed board prior mask."""
        self._feature_prior = prior_mask

    def detect(self, frame: np.ndarray, feature_prior: np.ndarray | None = None) -> bool:
        """Predict board mask using the model."""
        if self._model is None:
            return False

        h, w = frame.shape[:2]

        # Normalize frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
        img_tensor = img_tensor.to(self._device)

        # Predict
        with torch.no_grad():
            pred = self._model(img_tensor)

        # Handle dict output from models like DeepLabV3
        if isinstance(pred, dict):
            pred_logits = pred['out']
            pred = torch.sigmoid(pred_logits)

        # Convert to mask
        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()  # (H,W)
        pred_np = cv2.resize(pred_np, (w, h), interpolation=cv2.INTER_LINEAR)

        prior_map = self._spatial_prior.get_prior_map((h, w))
        if prior_map is not None:
            # Use the learned board prior as a spatial bias so the model keeps
            # board-shaped regions even when the raw probability map is weak.
            pred_np = np.clip(pred_np * (0.60 + 0.80 * prior_map), 0.0, 1.0)

        # NOTE: external HockeyRink-derived feature priors are disabled.
        # The model will still use the learned spatial prior (`BoardSpatialPrior`),
        # but any externally supplied `feature_prior` will be ignored to avoid
        # degrading the learned detector.

        self._prob_map = pred_np
        self._confidence_score = float(np.percentile(pred_np, 99.5))
        mask = (pred_np > 0.22).astype(np.uint8) * 255
        raw_mask = mask.copy()
        raw_mask[:int(h * 0.25), :int(w * 0.35)] = 0

        # Post-process: wider horizontal close to bridge broken board segments
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        # Gentler open to avoid erasing valid board corners
        k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
        mask[:int(h * 0.25), :int(w * 0.35)] = 0

        # Rebuild the band from per-column support so the mask stays continuous.
        # This matches the geometry of boards better than connected-component pruning.
        support = mask > 0
        top_y = np.full(w, np.nan, dtype=np.float32)
        bot_y = np.full(w, np.nan, dtype=np.float32)
        min_col_thickness = max(8, int(h * 0.03))

        for col in range(w):
            ys = np.where(support[:, col])[0]
            if ys.size >= 3 and (ys[-1] - ys[0] + 1) >= min_col_thickness:
                top_y[col] = float(ys.min())
                bot_y[col] = float(ys.max())

        valid = ~np.isnan(top_y) & ~np.isnan(bot_y)
        if valid.sum() >= max(20, w // 20):
            valid_x = np.where(valid)[0]
            missing_x = np.where(~valid)[0]
            top_y[missing_x] = np.interp(missing_x, valid_x, top_y[valid])
            bot_y[missing_x] = np.interp(missing_x, valid_x, bot_y[valid])

            # Apply robust median filtering first to strip out high-frequency noise and outlier spikes
            top_y_med = median_filter(top_y, size=max(11, w // 18) | 1).astype(np.float32)
            bot_y_med = median_filter(bot_y, size=max(7, w // 24) | 1).astype(np.float32)

            # Fit 2nd-degree polynomial to the filtered bottom coordinate (perfectly smooth, parabolic curve fit)
            x_coords = np.arange(w, dtype=np.float32)
            coeffs_bot = np.polyfit(x_coords, bot_y_med, deg=2)
            bot_y_smooth = np.polyval(coeffs_bot, x_coords).astype(np.float32)

            # Project upward using the board height prior based on the smoothed bottom coordinate
            t_col = np.clip(bot_y_smooth / float(h), 0.0, 1.0)
            board_h = BOARD_HEIGHT_TOP_PX + t_col * (BOARD_HEIGHT_BOTTOM_PX - BOARD_HEIGHT_TOP_PX)
            top_y_raw = np.clip(bot_y_smooth - board_h, h * BOARD_MAX_TOP_FRAC, h - 1)

            # Fit 2nd-degree polynomial to the top coordinate to ensure top line is equally smooth and clean
            coeffs_top = np.polyfit(x_coords, top_y_raw, deg=2)
            top_y_smooth = np.polyval(coeffs_top, x_coords).astype(np.float32)

            top_y = np.clip(top_y_smooth, 0, h - 1).astype(np.int32)
            bot_y = np.clip(bot_y_smooth, top_y + 1, h - 1).astype(np.int32)

            row_idx = np.arange(h, dtype=np.int32)[:, None]
            band = (row_idx >= top_y[None, :]) & (row_idx <= bot_y[None, :])
            reconstructed = np.zeros_like(mask)
            reconstructed[band] = 255

            # Fill small interior holes caused by logos, lettering, or trim gaps.
            flood = reconstructed.copy()
            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            cv2.floodFill(flood, flood_mask, (0, 0), 255)
            holes = cv2.bitwise_not(flood)
            reconstructed = cv2.bitwise_or(reconstructed, holes)

            reconstructed = cv2.morphologyEx(
                reconstructed,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7)),
            )

            # Preserve any raw support that survived thresholding but was not
            # captured by the reconstructed band.
            reconstructed = cv2.bitwise_or(reconstructed, raw_mask)

            # One final band-level close to merge adjacent board panels and
            # smooth out gaps left by logos or jersey-colored advertisements.
            reconstructed = cv2.morphologyEx(
                reconstructed,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (51, 9)),
            )

            recon_cov = float((reconstructed > 0).mean())
            if 0.005 <= recon_cov <= 0.25:
                mask = reconstructed
            else:
                mask = raw_mask
        else:
            mask = raw_mask

        self._board_mask = mask
        return True

    def get_board_mask(self) -> np.ndarray | None:
        """Return the predicted board mask."""
        return self._board_mask

    def get_probability_map(self) -> np.ndarray | None:
        """Return the latest resized probability map."""
        return self._prob_map

    def get_confidence_score(self) -> float:
        """Return a scalar confidence score for the latest prediction."""
        return self._confidence_score

    def is_ready(self) -> bool:
        """Returns True if model is loaded."""
        return self._model is not None
