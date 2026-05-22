#!/usr/bin/env python3
import json

NOTEBOOK_PATH = 'train_board_segmentation_colab.ipynb'

with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        modified = False
        for i, line in enumerate(source):
            # Fix directory path
            if line.startswith("ANN_DIR = 'colab_training_data/annotation_frames/new_batch/annotated'"):
                source[i] = "ANN_DIR = 'annotation_frames/new_batch/annotated'\n"
                modified = True
        
        if modified:
            # Add the 100% board filter to both loops
            # We will just replace the appending logic
            new_source = []
            for line in source:
                if 'pairs.append((orig, mask))' in line:
                    indent = line[:len(line) - len(line.lstrip())]
                    new_source.append(f'{indent}board_ratio = (mask > 0).mean()\n')
                    new_source.append(f'{indent}if board_ratio > 0.99:\n')
                    new_source.append(f'{indent}    print(f"  SKIP {{base.split(\'/\')[-1]}}: {{board_ratio:.1%}} board (too much)")\n')
                    new_source.append(f'{indent}    continue\n')
                    new_source.append(line)
                else:
                    new_source.append(line)
            cell['source'] = new_source

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("Fixed ANN_DIR and added 100% board filter to Colab notebook.")
