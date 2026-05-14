#!/usr/bin/env python3
"""
Train the board segmentation U-Net using manually-annotated frames.

Ground-truth masks are generated from the annotated PNGs (red top line,
yellow bottom line) via AnnotationBoardDetector.

Usage:
    python3 train_board_segmentation.py

Outputs:
    src/calibration/board_segmentation_model.pth  (updated weights)
    annotation_frames/training_debug/              (visualizations)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from pathlib import Path

from src.calibration.annotation_board_detector import AnnotationBoardDetector
from src.calibration.ml_board_detector import UNet, TARGET_WIDTH, TARGET_HEIGHT

ANNOTATION_DIR   = Path('annotation_frames')
DEBUG_DIR        = ANNOTATION_DIR / 'training_debug'
MODEL_PATH       = Path('src/calibration/board_segmentation_model.pth')
CONFIG_PATH      = Path('src/calibration/board_segmentation_model_config.json')

DEBUG_DIR.mkdir(exist_ok=True)

# Training hyperparameters
EPOCHS         = 80
LR             = 5e-4
AUGMENT_FACTOR = 12   # synthetic augmentation multiplier per real frame
BATCH_SIZE     = 4


# ── Dataset ────────────────────────────────────────────────────────────────────

class BoardDataset(Dataset):
    """Generates (image, mask) pairs from annotated PNG files."""

    def __init__(self, pairs, augment=True):
        self.pairs   = pairs   # list of (orig_bgr, mask_uint8)
        self.augment = augment
        self.items   = self._build_items()

    def _build_items(self):
        items = []
        for orig, mask in self.pairs:
            items.append((orig, mask))
            if self.augment:
                for _ in range(AUGMENT_FACTOR - 1):
                    items.append(self._augment(orig, mask))
        return items

    @staticmethod
    def _augment(img, mask):
        h, w = img.shape[:2]
        aug_img, aug_mask = img.copy(), mask.copy()

        # Horizontal flip
        if np.random.rand() > 0.5:
            aug_img  = cv2.flip(aug_img,  1)
            aug_mask = cv2.flip(aug_mask, 1)

        # Brightness / contrast
        alpha = np.random.uniform(0.75, 1.25)
        beta  = np.random.randint(-25, 25)
        aug_img = np.clip(aug_img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        # Hue/saturation jitter
        hsv = cv2.cvtColor(aug_img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-8, 8)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + np.random.randint(-20, 20), 0, 255)
        aug_img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Small scale crop
        scale = np.random.uniform(0.88, 1.0)
        new_h, new_w = int(h * scale), int(w * scale)
        y0 = np.random.randint(0, h - new_h + 1)
        x0 = np.random.randint(0, w - new_w + 1)
        aug_img  = cv2.resize(aug_img [y0:y0+new_h, x0:x0+new_w], (w, h))
        aug_mask = cv2.resize(aug_mask[y0:y0+new_h, x0:x0+new_w], (w, h),
                              interpolation=cv2.INTER_NEAREST)

        return aug_img, aug_mask

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img, mask = self.items[idx]
        img_small  = cv2.resize(img,  (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)

        img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0   # (3, H, W)
        y = torch.from_numpy((mask_small > 0).astype(np.float32)).unsqueeze(0)  # (1, H, W)
        return x, y


# ── Data loading ───────────────────────────────────────────────────────────────

FRAME_PAIRS = [
    ('v1_f010_overhead_end_zone.jpg', 'v1_f010_overhead_end_zone.png'),
    ('v1_f317_overhead_end_zone.jpg', 'v1_f317_overhead_end_zone.png'),
    ('v1_f634_overhead_end_zone.jpg', 'v1_f634_overhead_end_zone.png'),
    ('v3_f010_overhead_full.jpg',     'v3_f010_overhead_full.png'),
    ('v3_f200_overhead_full.jpg',     'v3_f200_overhead_full.png'),
    ('v4_f010_overhead_near.jpg',     'v4_f010_overhead_near.png'),
]

detector = AnnotationBoardDetector()
pairs    = []

print("Generating ground-truth masks from annotations...")
for orig_name, ann_name in FRAME_PAIRS:
    orig_path = ANNOTATION_DIR / orig_name
    ann_path  = ANNOTATION_DIR / ann_name

    orig = cv2.imread(str(orig_path))
    ann  = cv2.imread(str(ann_path))

    if orig is None:
        print(f"  ⚠ Missing original: {orig_path}"); continue
    if ann is None:
        print(f"  ⚠ Missing annotation: {ann_path}"); continue

    # Generate mask from annotation lines
    if not detector.detect(ann):
        print(f"  ⚠ No annotation lines found in {ann_name}"); continue

    mask = detector.get_board_mask()
    if mask is None:
        print(f"  ⚠ No mask generated from {ann_name}"); continue

    pairs.append((orig, mask))
    print(f"  ✓ {orig_name}: {(mask > 0).mean():.1%} of frame is board")

    # Save debug visualization
    debug = orig.copy()
    debug[mask > 0] = (debug[mask > 0] * 0.45 + np.array([0, 200, 0]) * 0.55).astype(np.uint8)
    cv2.imwrite(str(DEBUG_DIR / orig_name.replace('.jpg', '_gt.jpg')), debug)

if not pairs:
    print("ERROR: No training pairs generated. Check annotation files.")
    exit(1)

print(f"\nTraining on {len(pairs)} annotated frames × {AUGMENT_FACTOR} augmentations = "
      f"{len(pairs) * AUGMENT_FACTOR} samples")


# ── Training ───────────────────────────────────────────────────────────────────

dataset    = BoardDataset(pairs, augment=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = UNet(in_channels=3, out_channels=1).to(device)

# Load existing weights as starting point if available
if MODEL_PATH.exists():
    try:
        model.load_state_dict(torch.load(str(MODEL_PATH), map_location=device))
        print(f"Loaded existing weights from {MODEL_PATH} (fine-tuning)")
    except Exception as e:
        print(f"Could not load existing model ({e}), training from scratch")

# Weighted BCE: penalise false negatives more than false positives
# (Better to include a little extra than to miss the boards)
pos_weight = torch.tensor([2.5]).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Use raw logits version — swap sigmoid out of model for training
class UNetLogits(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        e1 = self.base.enc1(x)
        e2 = self.base.enc2(self.base.pool1(e1))
        e3 = self.base.enc3(self.base.pool2(e2))
        b  = self.base.bottleneck(self.base.pool3(e3))
        d3 = self.base.dec3(torch.cat([self.base.upconv3(b), e3], dim=1))
        d2 = self.base.dec2(torch.cat([self.base.upconv2(d3), e2], dim=1))
        d1 = self.base.dec1(torch.cat([self.base.upconv1(d2), e1], dim=1))
        return self.base.final(d1)   # raw logits

model_logits = UNetLogits(model).to(device)
optimizer    = optim.Adam(model_logits.parameters(), lr=LR, weight_decay=1e-5)
scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.05)

best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    model_logits.train()
    epoch_loss = 0.0
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        logits = model_logits(x_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_logits.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / len(dataloader)

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  loss={avg_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), str(MODEL_PATH))

print(f"\nBest loss: {best_loss:.4f}")
print(f"Model saved → {MODEL_PATH}")

# Save config
config = {
    "target_width":  TARGET_WIDTH,
    "target_height": TARGET_HEIGHT,
    "threshold":     0.18,
    "trained_on_frames": [p[0] for p in FRAME_PAIRS],
    "epochs": EPOCHS,
    "best_loss": float(best_loss),
}
with open(str(CONFIG_PATH), 'w') as f:
    json.dump(config, f, indent=2)
print(f"Config saved → {CONFIG_PATH}")


# ── Final validation visualization ────────────────────────────────────────────
print("\nGenerating validation overlays...")
model.eval()

for orig_name, ann_name in FRAME_PAIRS:
    orig = cv2.imread(str(ANNOTATION_DIR / orig_name))
    if orig is None: continue
    h, w = orig.shape[:2]

    img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
    x   = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x   = x.to(device)

    with torch.no_grad():
        pred = model(x)

    prob = pred.squeeze().cpu().numpy()
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = (prob > 0.18).astype(np.uint8) * 255

    overlay = orig.copy()
    overlay[mask > 0] = (overlay[mask > 0] * 0.45 + np.array([0, 200, 0]) * 0.55).astype(np.uint8)
    out_path = str(DEBUG_DIR / orig_name.replace('.jpg', '_pred.jpg'))
    cv2.imwrite(out_path, overlay)
    print(f"  {out_path}")

print("\nDone. Check annotation_frames/training_debug/ for GT and prediction overlays.")
