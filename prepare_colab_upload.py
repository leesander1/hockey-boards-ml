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

import glob

ANNOTATION_FRAMES = []
ANN_DIR = 'annotation_frames/new_batch'
ANNOTATED_DIR = f'{ANN_DIR}/annotated'

PSEUDO_DIR = 'annotation_frames/pseudo_labeled'

# Find all annotated and mask images
mask_files = glob.glob(f'{ANNOTATED_DIR}/*-mask.png') + glob.glob(f'{PSEUDO_DIR}/*-mask.png')
annotated_files = glob.glob(f'{ANNOTATED_DIR}/*-annotated.png') + glob.glob(f'{ANNOTATED_DIR}/*-annotate.png') + glob.glob(f'{PSEUDO_DIR}/*-annotated.png')

bases = set()
for m in mask_files:
    bases.add(m.replace('-mask.png', ''))
for a in annotated_files:
    bases.add(a.replace('-annotated.png', '').replace('-annotate.png', ''))

ANNOTATION_FRAMES_ALL = []
for base in bases:
    orig_path = f'{base}.jpg'
    if os.path.exists(orig_path):
        ANNOTATION_FRAMES_ALL.append(orig_path)
        if f'{base}-mask.png' in mask_files:
            ANNOTATION_FRAMES_ALL.append(f'{base}-mask.png')
        # check both annotated and annotate
        if f'{base}-annotated.png' in annotated_files:
            ANNOTATION_FRAMES_ALL.append(f'{base}-annotated.png')
        if f'{base}-annotate.png' in annotated_files:
            ANNOTATION_FRAMES_ALL.append(f'{base}-annotate.png')

MODEL_PATH = 'src/calibration/board_segmentation_model.pth'

# Script paths for warp-to-template training
WARP_SCRIPT = 'scripts/warp_to_template.py'
TRAIN_WARPED_SCRIPT = 'scripts/train_board_segmentation_warped.py'
WARPED_DATASET = 'src/data/warped_dataset.py'

KEYPOINT_MAP_TEMPLATE_PATH = 'hockeyrink_keypoint_map.template.json'

KEYPOINT_MAP_CANDIDATES = [
    os.environ.get('HOCKEYRINK_KEYPOINT_MAP_PATH'),
    'hockeyrink_keypoint_map.json',
    'hockeyrink_keypoint_world_map.json',
    'annotation_frames/new_batch/hockeyrink_keypoint_map.json',
]


def _find_keypoint_map_path() -> str | None:
    for candidate in KEYPOINT_MAP_CANDIDATES:
        if candidate and os.path.exists(candidate):
            return candidate
    return None

print(f'Packaging Colab training data → {OUT_ZIP}')
total = 0
keypoint_map_path = _find_keypoint_map_path()

with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    # Annotation frames
    for path in ANNOTATION_FRAMES_ALL:
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

    # Warp-to-template training scripts
    # Package all Python source code in src/ and scripts/
    script_files = []
    import glob
    script_files.extend(glob.glob('src/**/*.py', recursive=True))
    script_files.extend(glob.glob('scripts/**/*.py', recursive=True))
    for script_path in script_files:
        if os.path.exists(script_path):
            zf.write(script_path, script_path)
            sz = os.path.getsize(script_path)
            total += sz
            print(f'  + {script_path}  ({sz/1024:.0f} KB)')
    # Optional HockeyRink keypoint-to-world map for runtime geometry anchoring.
    if keypoint_map_path is not None:
        zf.write(keypoint_map_path, 'hockeyrink_keypoint_map.json')
        sz = os.path.getsize(keypoint_map_path)
        total += sz
        print(f'  + hockeyrink_keypoint_map.json  ({sz/1024:.0f} KB)')
    else:
        print('  ℹ No HockeyRink keypoint map found — Colab runtime will need one only if using HockeyRink anchors')

    if os.path.exists(KEYPOINT_MAP_TEMPLATE_PATH):
        zf.write(KEYPOINT_MAP_TEMPLATE_PATH, 'hockeyrink_keypoint_map.template.json')
        sz = os.path.getsize(KEYPOINT_MAP_TEMPLATE_PATH)
        total += sz
        print(f'  + hockeyrink_keypoint_map.template.json  ({sz/1024:.0f} KB)')

zip_size = os.path.getsize(OUT_ZIP)
print(f'\n✓ Done: {OUT_ZIP}  ({zip_size/1024/1024:.1f} MB)')
print('\nWorkflow 1: Standard training on original frames')
print('  1. Open train_board_segmentation_colab.ipynb in Google Colab')
print('  2. Runtime → Change runtime type → T4 GPU')
print('  3. Run all cells — upload the zip when prompted')
print('  4. Download board_segmentation_model.pth at the end')
print('  5. Replace src/calibration/board_segmentation_model.pth in this repo')
print('\nWorkflow 2: Warp-to-template training (better generalization)')
print('  1. Upload the zip to Colab')
print('  2. Run: python scripts/warp_to_template.py  (creates data/warped/train/)')
print('  3. Run: python scripts/train_board_segmentation_warped.py --epochs 50')
print('  4. Download models/board_segmentation_warped.pth')
print('  5. (Optional) Fine-tune on original frames for last few epochs')
