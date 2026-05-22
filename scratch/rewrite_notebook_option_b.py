#!/usr/bin/env python3
import json

NOTEBOOK_PATH = 'train_board_segmentation_colab.ipynb'

with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

# Update header
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        if 'Hockey Board Segmentation' in "".join(cell.get('source', [])):
            cell['source'] = [
                "# 🏒 Hockey Board Segmentation — Option B (Direct Inference)\n",
                "\n",
                "This notebook implements the optimal strategy for **Direct Inference** (Option B).\n",
                "Instead of warping frames at runtime, we train directly on the raw, angled broadcast frames.\n",
                "To ensure the model learns to generalize across different camera angles and rink perspectives,\n",
                "we use aggressive **Perspective Data Augmentations** (via Albumentations) during training.\n",
                "\n",
                "## Setup steps\n",
                "1. **Runtime → Change runtime type → T4 GPU** (free tier is enough)\n",
                "2. Upload the zip from your local machine: `python3 prepare_colab_upload.py` → uploads `colab_training_data.zip`\n",
                "3. Run all cells\n",
                "4. Download the output model and replace `src/calibration/board_segmentation_model.pth`"
            ]

# Inject Albumentations
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell.get('source', []))
        
        # 1. Update pip install
        if "!pip install -q opencv-python-headless" in source_str:
            cell['source'] = ["!pip install -q opencv-python-headless scipy albumentations\n"]
        
        # 2. Fix ANN_DIR and add 100% board filter, plus Pseudo Labels
        if "def extract_mask_from_annotation" in source_str and "ANN_DIR" in source_str:
            new_source = []
            for line in cell['source']:
                if line.startswith("ANN_DIR ="):
                    new_source.append("ANN_DIR = 'annotation_frames/new_batch/annotated'\n")
                    new_source.append("PSEUDO_DIR = 'annotation_frames/pseudo_labeled'\n")
                elif line.startswith("mask_files = "):
                    new_source.append("mask_files = glob.glob(f'{ANN_DIR}/*-mask.png') + glob.glob(f'{PSEUDO_DIR}/*-mask.png')\n")
                elif line.startswith("annotated_files = "):
                    new_source.append("annotated_files = glob.glob(f'{ANN_DIR}/*-annotated.png') + glob.glob(f'{ANN_DIR}/*-annotate.png') + glob.glob(f'{PSEUDO_DIR}/*-annotated.png')\n")
                elif 'pairs.append((orig, mask))' in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    new_source.append(f'{indent}board_ratio = (mask > 0).mean()\n')
                    new_source.append(f'{indent}if board_ratio > 0.99:\n')
                    new_source.append(f'{indent}    print(f"  SKIP {{base.split(\'/\')[-1]}}: {{board_ratio:.1%}} board (too much)")\n')
                    new_source.append(f'{indent}    continue\n')
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
        # 3. Rewrite BoardDataset to use Albumentations
        if "class BoardDataset(Dataset):" in source_str:
            dataset_code = """import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

EPOCHS         = 150
LR             = 5e-4
AUGMENT_FACTOR = 25
BATCH_SIZE     = 8

# Aggressive perspective and photometric augmentations
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    # Crucial for Option B: simulates different camera angles across the ice!
    A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.8),
    A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-5, 5), p=0.7),
    A.GridDistortion(p=0.4), # Added advanced distortion for lens curvature
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    A.GaussianBlur(blur_limit=(3, 7), p=0.5),
    A.GaussNoise(std_range=(0.1, 0.3), p=0.5),
    # Simulate players/refs blocking the boards
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(20, 80), hole_width_range=(20, 80), p=0.6),
    A.Resize(TARGET_HEIGHT, TARGET_WIDTH),
    A.Normalize(mean=(0,0,0), std=(1,1,1)), # Keep 0-1 scale like previous logic
    ToTensorV2()
])

class BoardDataset(Dataset):
    def __init__(self, pairs, augment=True):
        self.items = []
        for orig, mask in pairs:
            # We add the base unaugmented frame
            self.items.append((orig, mask, False))
            if augment:
                for _ in range(AUGMENT_FACTOR - 1):
                    self.items.append((orig, mask, True))

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        img, mask, apply_aug = self.items[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if apply_aug:
            augmented = train_transform(image=img, mask=mask)
            x = augmented['image']
            y = augmented['mask']
        else:
            # Just resize and convert
            img = cv2.resize(img, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)
            y = cv2.resize(mask, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_NEAREST)
            x = torch.from_numpy(img).float().permute(2,0,1) / 255.0
            y = torch.from_numpy(y)

        y = (y > 0).float().unsqueeze(0)
        return x, y

dataset    = BoardDataset(pairs, augment=True)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print(f'Dataset: {len(dataset)} samples ({len(pairs)} frames × {AUGMENT_FACTOR})')

# Logits model for training (skip sigmoid for BCEWithLogitsLoss)
class UNetLogits(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, x):
        e1 = self.base.enc1(x)
        e2 = self.base.enc2(self.base.pool1(e1))
        e3 = self.base.enc3(self.base.pool2(e2))
        b  = self.base.bottleneck(self.base.pool3(e3))
        
        # Safe upsampling (matching ml_board_detector.py robust logic)
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

model_logits = UNetLogits(model).to(device)
pos_weight   = torch.tensor([3.0]).to(device)

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

criterion    = DiceBCELoss(pos_weight=pos_weight)
optimizer    = optim.AdamW(model_logits.parameters(), lr=LR, weight_decay=1e-4)
scheduler    = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR*0.02)

losses = []
best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
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
    losses.append(avg)
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), 'board_segmentation_model.pth')
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{EPOCHS}  loss={avg:.4f}  best={best_loss:.4f}')

print(f'\\nTraining complete. Best loss: {best_loss:.4f}')
"""
            cell['source'] = [line + "\n" for line in dataset_code.split('\n')]

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("Rewrote notebook for Option B with Albumentations.")
