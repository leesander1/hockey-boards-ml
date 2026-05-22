import sys
import os
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

TARGET_WIDTH  = 640
TARGET_HEIGHT = 360

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = self._conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self._conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = self._conv_block(128, 256)
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
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.upconv3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.sigmoid(self.final(d1))

class UNetLogits(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        e1 = self.base.enc1(x)
        e2 = self.base.enc2(self.base.pool1(e1))
        e3 = self.base.enc3(self.base.pool2(e2))
        b  = self.base.bottleneck(self.base.pool3(e3))
        import torch.nn.functional as F
        up3 = self.base.upconv3(b)
        if up3.shape != e3.shape: up3 = F.interpolate(up3, size=e3.shape[2:])
        d3 = self.base.dec3(torch.cat([up3, e3], dim=1))
        
        up2 = self.base.upconv2(d3)
        if up2.shape != e2.shape: up2 = F.interpolate(up2, size=e2.shape[2:])
        d2 = self.base.dec2(torch.cat([up2, e2], dim=1))
        
        up1 = self.base.upconv1(d2)
        if up1.shape != e1.shape: up1 = F.interpolate(up1, size=e1.shape[2:])
        d1 = self.base.dec1(torch.cat([up1, e1], dim=1))
        return self.base.final(d1)

def extract_mask_from_annotation(ann_bgr, orig_bgr=None, is_mask=False):
    if is_mask:
        if len(ann_bgr.shape) == 3:
            ann_bgr = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2GRAY)
        return (ann_bgr > 0).astype(np.uint8) * 255
    else:
        if orig_bgr is not None:
            diff = cv2.absdiff(ann_bgr, orig_bgr)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            mask = (diff_gray > 20).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
            return mask
        else:
            hsv = cv2.cvtColor(ann_bgr, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            return green_mask

def load_data(base_dir):
    ANN_DIR = f'{base_dir}/annotation_frames/new_batch/annotated'
    PSEUDO_DIR = f'{base_dir}/annotation_frames/pseudo_labeled'
    pairs = []
    processed_bases = set()
    mask_files = glob.glob(f'{ANN_DIR}/*-mask.png') + glob.glob(f'{PSEUDO_DIR}/*-mask.png')
    annotated_files = glob.glob(f'{ANN_DIR}/*-annotated.png') + glob.glob(f'{ANN_DIR}/*-annotate.png') + glob.glob(f'{PSEUDO_DIR}/*-annotated.png')
    
    for m_path in mask_files:
        base = m_path.replace('-mask.png', '')
        orig_path = f'{base}.jpg'
        orig = cv2.imread(orig_path)
        mask_img = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if orig is None or mask_img is None: continue
        mask = extract_mask_from_annotation(mask_img, is_mask=True)
        board_ratio = (mask > 0).mean()
        if board_ratio > 0.99: continue
        pairs.append((orig, mask))
        processed_bases.add(base)

    for a_path in annotated_files:
        base = a_path.replace('-annotated.png', '').replace('-annotate.png', '')
        if base in processed_bases: continue
        orig_path = f'{base}.jpg'
        orig = cv2.imread(orig_path)
        ann = cv2.imread(a_path)
        if orig is None or ann is None: continue
        mask = extract_mask_from_annotation(ann, orig_bgr=orig, is_mask=False)
        board_ratio = (mask > 0).mean()
        if board_ratio > 0.99: continue
        pairs.append((orig, mask))
        processed_bases.add(base)
    return pairs

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.8),
    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-5, 5), p=0.7),
    A.GridDistortion(p=0.4),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.GaussNoise(std_range=(0.1, 0.3), p=0.5),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(20, 80), hole_width_range=(20, 80), p=0.6),
    A.Normalize(mean=(0,0,0), std=(1,1,1)),
    ToTensorV2()
])

class BoardDataset(Dataset):
    def __init__(self, pairs, augment=True, augment_factor=5):
        self.augment = augment
        self.augment_factor = augment_factor
        self.resized_pairs = []
        for orig, mask in pairs:
            img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            mask_resized = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
            self.resized_pairs.append((img_resized, mask_resized))
            
    def __len__(self):
        if self.augment:
            return len(self.resized_pairs) * self.augment_factor
        return len(self.resized_pairs)

    def __getitem__(self, idx):
        real_idx = idx % len(self.resized_pairs)
        img, mask = self.resized_pairs[real_idx]
        is_base = (idx < len(self.resized_pairs))
        if self.augment and not is_base:
            augmented = train_transform(image=img, mask=mask)
            x = augmented['image']
            y = augmented['mask']
        else:
            x = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            y = torch.from_numpy(mask).float()
        y = (y > 0).float().unsqueeze(0)
        return x, y

class DiceBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets, smooth=1):
        inputs_sig = torch.sigmoid(inputs)
        inputs_flat = inputs_sig.view(-1)
        targets_flat = targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs_flat.sum() + targets_flat.sum() + smooth)
        bce_loss = self.bce(inputs, targets)
        return bce_loss + dice_loss

def main():
    base_dir = 'scratch/training_data/colab_training_data'
    pairs = load_data(base_dir)
    print(f"Loaded {len(pairs)} frames")
    if not pairs:
        print("No training data found!")
        return

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    model = UNet().to(device)
    model_logits = UNetLogits(model).to(device)
    
    dataset = BoardDataset(pairs, augment=True, augment_factor=5)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    
    pos_weight = torch.tensor([3.0]).to(device)
    criterion = DiceBCELoss(pos_weight=pos_weight)
    
    LR = 5e-4
    EPOCHS = 100
    optimizer = optim.AdamW(model_logits.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.02)
    
    best_loss = float('inf')
    
    # Fast training loop
    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()
        model_logits.train()
        ep_loss = 0.0
        for x_b, y_b in dataloader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model_logits(x_b), y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_logits.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item()
        scheduler.step()
        avg = ep_loss / len(dataloader)
        elapsed = time.time() - start_time
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), 'src/calibration/board_segmentation_model_unet.pth')
            # Update the fallback copy as well just in case
            torch.save(model.state_dict(), 'src/calibration/board_segmentation_model.pth')
        print(f'Epoch {epoch:3d}/{EPOCHS} | loss={avg:.4f} | best={best_loss:.4f} | time={elapsed:.2f}s')

if __name__ == "__main__":
    main()
