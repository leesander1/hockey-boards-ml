#!/usr/bin/env python3
"""Train a U-Net for board segmentation using annotated frames as ground truth.

Uses the red (top) and yellow (bottom) annotation lines to generate perfect masks,
then trains a semantic segmentation model to predict board regions.

Usage: python3 scripts/train_board_segmentation.py
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from glob import glob

ANN_DIR = os.path.join(os.path.dirname(__file__), '..', 'annotation_frames')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'calibration', 'board_segmentation_model.pth')
TARGET_WIDTH = 640
TARGET_HEIGHT = 360
SCALE_VARIANTS = (
    (1.00, 1.00),
    (1.08, 1.00),
    (0.92, 1.00),
    (1.00, 0.92),
)


class BoardAnnotationDataset(Dataset):
    """Dataset that loads source frames and derives masks from matching annotations."""

    def __init__(self, annotation_dir):
        self.samples = []

        # Primary training set: source JPGs with matching annotation PNGs.
        for image_path in sorted(glob(os.path.join(annotation_dir, '*.jpg'))):
            stem = os.path.splitext(os.path.basename(image_path))[0]
            annotation_path = os.path.join(annotation_dir, f'{stem}.png')
            if os.path.exists(annotation_path):
                self.samples.append((image_path, annotation_path))

        self.variant_count = len(SCALE_VARIANTS) + 1

    def __len__(self):
        return len(self.samples) * self.variant_count

    def __getitem__(self, idx):
        sample_idx = idx // self.variant_count
        variant_idx = idx % self.variant_count
        image_path, annotation_path = self.samples[sample_idx]

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f'failed to load {image_path}')

        # Convert to RGB (from BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if variant_idx > 0:
            img, annotation = self._augment_pair(img, annotation_path, variant_idx)
        else:
            annotation = cv2.imread(annotation_path)
            if annotation is None:
                raise ValueError(f'failed to load {annotation_path}')

        annotation = cv2.cvtColor(annotation, cv2.COLOR_BGR2HSV)
        # Extract ground-truth mask from the matching annotation overlay.
        hsv = annotation

        h, w = img.shape[:2]

        # Red (top)
        red = cv2.inRange(hsv, np.array([0, 150, 150], np.uint8), np.array([10, 255, 255], np.uint8))
        red = cv2.bitwise_or(red, cv2.inRange(hsv, np.array([170, 150, 150], np.uint8), np.array([180, 255, 255], np.uint8)))
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

        # Yellow (bottom)
        yellow = cv2.inRange(hsv, np.array([18, 150, 150], np.uint8), np.array([35, 255, 255], np.uint8))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 3)))

        # Per-column extremes
        red_top_y = np.full(w, np.nan, dtype=np.float32)
        yellow_bot_y = np.full(w, np.nan, dtype=np.float32)

        for col in range(w):
            r_ys = np.where(red[:, col] > 0)[0]
            y_ys = np.where(yellow[:, col] > 0)[0]
            if r_ys.size:
                red_top_y[col] = float(r_ys.min())
            if y_ys.size:
                yellow_bot_y[col] = float(y_ys.max())

        # Interpolate NaNs
        valid = ~np.isnan(red_top_y)
        if valid.sum() > 0:
            valid_x = np.where(valid)[0]
            red_top_y[~valid] = np.interp(np.arange(w)[~valid], valid_x, red_top_y[valid])

        valid = ~np.isnan(yellow_bot_y)
        if valid.sum() > 0:
            valid_x = np.where(valid)[0]
            yellow_bot_y[~valid] = np.interp(np.arange(w)[~valid], valid_x, yellow_bot_y[valid])

        # Build mask.
        mask = np.zeros((h, w), dtype=np.uint8)
        row_idx = np.arange(h, dtype=np.int32)[:, None]
        red_top_y_int = np.clip(red_top_y, 0, h - 1).astype(np.int32)
        yellow_bot_y_int = np.clip(yellow_bot_y, 0, h - 1).astype(np.int32)
        in_board = (row_idx >= red_top_y_int[None, :]) & (row_idx <= yellow_bot_y_int[None, :])
        mask[in_board] = 255

        # Exclude scorebug zone.
        mask[:int(h * 0.25), :int(w * 0.35)] = 0

        # Resize to a fixed training resolution to stabilize batching and learning.
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)

        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor

    def _augment_pair(self, img, annotation_path, variant_idx):
        """Apply a deterministic image augmentation to a paired sample."""
        annotation = cv2.imread(annotation_path)
        if annotation is None:
            raise ValueError(f'failed to load {annotation_path}')

        if variant_idx == 1:
            img = cv2.flip(img, 1)
            annotation = cv2.flip(annotation, 1)
        elif variant_idx == 2:
            img = cv2.convertScaleAbs(img, alpha=1.08, beta=8)
        elif variant_idx == 3:
            img = cv2.convertScaleAbs(img, alpha=0.92, beta=-6)

        return img, annotation


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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
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
        d3 = self.dec3(torch.cat([self.upconv3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.sigmoid(self.final(d1))


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Dataset and loader
    dataset = BoardAnnotationDataset(ANN_DIR)
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    print(f'Loaded {len(dataset)} paired annotation samples')

    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # Training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, mask in loader:
            img = img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            pred = model(img)
            bce = criterion(pred, mask)
            intersection = (pred * mask).sum(dim=(1, 2, 3))
            dice = 1.0 - ((2.0 * intersection + 1.0) / (pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1.0))
            loss = bce + dice.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}')

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}')

    # Also save hyperparams
    config_path = MODEL_PATH.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump({'in_channels': 3, 'out_channels': 1, 'device': device.type}, f)


if __name__ == '__main__':
    train()
