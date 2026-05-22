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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import json
from glob import glob

ANN_DIR = os.path.join(os.path.dirname(__file__), '..', 'annotation_frames')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'src', 'calibration', 'board_segmentation_model.pth')
TARGET_WIDTH = 320
TARGET_HEIGHT = 176


class BoardAnnotationDataset(Dataset):
    """Dataset that preloads and caches source frames and pre-straightened masks in RAM to avoid slow disk I/O."""

    def __init__(self, annotation_dir):
        self.samples = []

        # Recursively find annotated PNGs and map to matching JPGs and straightened masks
        annotated_paths = sorted(
            glob(os.path.join(annotation_dir, '**', '*-annotated.png'), recursive=True) +
            glob(os.path.join(annotation_dir, '**', '*-annotate.png'), recursive=True)
        )

        print("Preloading and caching all images and masks in RAM...", flush=True)
        for ann_path in annotated_paths:
            stem = os.path.basename(ann_path).replace('-annotated.png', '').replace('-annotate.png', '')
            image_path = os.path.join(os.path.dirname(ann_path), f"{stem}.jpg")
            mask_path = os.path.join(os.path.dirname(ann_path), f"{stem}-mask.png")
            if os.path.exists(image_path) and os.path.exists(mask_path):
                img = cv2.imread(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if img is not None and mask is not None:
                    # Convert to RGB (from BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.samples.append((img, mask))
                else:
                    print(f"Warning: Failed to load {image_path} or {mask_path}", flush=True)

        self.variant_count = 2
        print(f"Finished caching {len(self.samples)} original pairs into RAM. Total training size: {len(self.samples) * self.variant_count}", flush=True)

    def __len__(self):
        return len(self.samples) * self.variant_count

    def __getitem__(self, idx):
        sample_idx = idx // self.variant_count
        variant_idx = idx % self.variant_count
        orig_img, orig_mask = self.samples[sample_idx]

        # Copy arrays so we don't mutate the cached originals
        img = orig_img.copy()
        mask = orig_mask.copy()
        
        # Apply flip augmentation
        if variant_idx == 1:
            # Flip image and mask horizontally
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # Resize to training resolution
        img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)

        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)

        mask_tensor = torch.from_numpy(mask).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)

        return img_tensor, mask_tensor


class UNet(nn.Module):
    """Lightweight U-Net for board segmentation matching production exactly."""

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


def train():
    # Force CPU to avoid Apple Silicon MPS deadlocks and kernel bugs
    device = torch.device('cpu')
    print(f'Using device: {device}', flush=True)

    # Maximize CPU usage
    torch.set_num_threads(8)
    print(f'Using {torch.get_num_threads()} CPU threads', flush=True)

    # Dataset and loader
    dataset = BoardAnnotationDataset(ANN_DIR)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Model
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    # Training loop
    epochs = 4
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
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}', flush=True)

    # Save
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Saved model to {MODEL_PATH}', flush=True)

    # Also save hyperparams
    config_path = MODEL_PATH.replace('.pth', '_config.json')
    with open(config_path, 'w') as f:
        json.dump({'in_channels': 3, 'out_channels': 1, 'device': device.type, 'architecture': 'unet'}, f)


if __name__ == '__main__':
    train()
