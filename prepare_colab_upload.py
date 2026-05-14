#!/usr/bin/env python3
"""
Packages the annotation frames + pre-trained model into a zip for Colab upload.

Usage:
    python3 prepare_colab_upload.py

Output:
    colab_training_data.zip   (~30-50 MB, upload this to Colab)
"""

import zipfile, os, shutil
from pathlib import Path

OUT_ZIP = 'colab_training_data.zip'

ANNOTATION_FRAMES = [
    # (original jpg, annotated png)
    ('annotation_frames/v1_f010_overhead_end_zone.jpg', 'annotation_frames/v1_f010_overhead_end_zone.png'),
    ('annotation_frames/v1_f317_overhead_end_zone.jpg', 'annotation_frames/v1_f317_overhead_end_zone.png'),
    ('annotation_frames/v1_f634_overhead_end_zone.jpg', 'annotation_frames/v1_f634_overhead_end_zone.png'),
    ('annotation_frames/v3_f010_overhead_full.jpg',     'annotation_frames/v3_f010_overhead_full.png'),
    ('annotation_frames/v3_f200_overhead_full.jpg',     'annotation_frames/v3_f200_overhead_full.png'),
    ('annotation_frames/v4_f010_overhead_near.jpg',     'annotation_frames/v4_f010_overhead_near.png'),
]

MODEL_PATH = 'src/calibration/board_segmentation_model.pth'

print(f'Packaging Colab training data → {OUT_ZIP}')
total = 0

with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    # Annotation frames
    for orig, ann in ANNOTATION_FRAMES:
        for path in (orig, ann):
            if os.path.exists(path):
                arcname = path   # keep directory structure
                zf.write(path, arcname)
                sz = os.path.getsize(path)
                total += sz
                print(f'  + {arcname}  ({sz/1024:.0f} KB)')
            else:
                print(f'  ⚠ MISSING: {path}')

    # Pre-trained model weights (starting point for fine-tuning)
    if os.path.exists(MODEL_PATH):
        zf.write(MODEL_PATH, 'board_segmentation_model.pth')
        sz = os.path.getsize(MODEL_PATH)
        total += sz
        print(f'  + board_segmentation_model.pth  ({sz/1024/1024:.1f} MB)')
    else:
        print(f'  ℹ No pre-trained model found — Colab will train from scratch')

zip_size = os.path.getsize(OUT_ZIP)
print(f'\n✓ Done: {OUT_ZIP}  ({zip_size/1024/1024:.1f} MB)')
print('\nNext steps:')
print('  1. Open train_board_segmentation_colab.ipynb in Google Colab')
print('  2. Runtime → Change runtime type → T4 GPU')
print('  3. Run all cells — upload the zip when prompted')
print('  4. Download board_segmentation_model.pth at the end')
print('  5. Replace src/calibration/board_segmentation_model.pth in this repo')
