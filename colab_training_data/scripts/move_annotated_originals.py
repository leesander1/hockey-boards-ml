#!/usr/bin/env python3
"""
Moves the original `.jpg` frames from `unannotated/` to `annotated/`
for any frames that have been annotated.
"""

import os
import shutil
import glob

ANN_DIR = 'annotation_frames/new_batch'
ANNOTATED_DIR = f'{ANN_DIR}/annotated'
UNANNOTATED_DIR = f'{ANN_DIR}/unannotated'

# Find all annotated and mask files
mask_files = glob.glob(f'{ANNOTATED_DIR}/*-mask.png')
annotated_files = glob.glob(f'{ANNOTATED_DIR}/*-annotated.png') + glob.glob(f'{ANNOTATED_DIR}/*-annotate.png')

# Extract base names (e.g. "2026-05-19_23-23-08_f0220")
bases = set()
for f in mask_files:
    basename = os.path.basename(f).replace('-mask.png', '')
    bases.add(basename)
for f in annotated_files:
    basename = os.path.basename(f).replace('-annotated.png', '').replace('-annotate.png', '')
    bases.add(basename)

moved_count = 0
missing_count = 0

print(f"Checking {len(bases)} annotated base frames...")

for base in sorted(bases):
    dest_jpg = os.path.join(ANNOTATED_DIR, f"{base}.jpg")
    
    # Check if the original JPG already exists in the annotated directory
    if os.path.exists(dest_jpg):
        continue
        
    # Search for the original JPG in the unannotated directory
    src_jpg = os.path.join(UNANNOTATED_DIR, f"{base}.jpg")
    if os.path.exists(src_jpg):
        shutil.move(src_jpg, dest_jpg)
        print(f"Moved original: {base}.jpg  →  annotated/")
        moved_count += 1
    else:
        print(f"⚠ Warning: Original frame not found for {base} in {UNANNOTATED_DIR}")
        missing_count += 1

print(f"\nSummary:")
print(f"  Moved: {moved_count} original frames")
if missing_count > 0:
    print(f"  Missing: {missing_count} original frames")
else:
    print(f"  All annotated frames have their matching original .jpg files!")
