#!/usr/bin/env python3
"""Train board segmentation model on warped (template-space) data.

This script:
1. Loads warped training data from data/warped/train/.
2. Trains a UNet segmentation model using combined Dice + BCE loss.
3. Validates on held-out warped frames.
4. Saves the trained model.

Run on Colab:
    python scripts/train_board_segmentation_warped.py --epochs 50 --batch-size 8 --lr 1e-3
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm

from src.calibration.ml_board_detector import UNet
from src.data.warped_dataset import build_dataloaders


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


class CombinedLoss(nn.Module):
    """Combined Dice + BCE loss."""
    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCELoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice_loss = self.dice(pred, target)
        bce_loss = self.bce(pred, target)
        return self.dice_weight * dice_loss + self.bce_weight * bce_loss


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(train_loader, desc='Training', leave=False):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc='Validating', leave=False):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            # Compute IoU
            pred_binary = (outputs > 0.5).float()
            intersection = (pred_binary * masks).sum(dim=(1,2,3))
            union = ((pred_binary + masks) > 0).float().sum(dim=(1,2,3))
            iou = (intersection / (union + 1e-6)).mean().item()
            total_iou += iou
    
    return total_loss / len(val_loader), total_iou / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description='Train board segmentation on warped data.')
    parser.add_argument('--data-dir', default='data/warped', help='Data directory.')
    parser.add_argument('--output-dir', default='models', help='Output directory for checkpoints.')
    parser.add_argument('--model-name', default='board_segmentation_warped.pth', help='Model filename.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers.')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device.')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print(f"Training on {device}")
    print(f"Data directory: {args.data_dir}")
    
    # Build dataloaders
    try:
        train_loader, val_loader = build_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except RuntimeError as e:
        print(f"Error: {e}")
        print("Please run: python scripts/warp_to_template.py")
        return 1
    
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Build model
    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = CombinedLoss(dice_weight=0.5, bce_weight=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    best_val_iou = 0.0
    best_model_path = os.path.join(args.output_dir, args.model_name)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Saved best model to {best_model_path}")
    
    print(f"\nTraining complete! Best Val IoU: {best_val_iou:.4f}")
    print(f"Model saved to: {best_model_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
