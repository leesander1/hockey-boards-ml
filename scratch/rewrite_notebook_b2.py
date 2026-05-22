#!/usr/bin/env python3
import json

NOTEBOOK_PATH = 'train_board_segmentation_colab.ipynb'

with open(NOTEBOOK_PATH, 'r') as f:
    nb = json.load(f)

old_cells = nb['cells']

# Find specific code cells by content
def find_code_cell(snippet):
    for cell in old_cells:
        if cell['cell_type'] == 'code':
            source_str = "".join(cell.get('source', []))
            if snippet in source_str:
                return cell.copy()
    raise ValueError(f"Cell with snippet '{snippet}' not found")

cell_upload = find_code_cell("print('Upload colab_training_data.zip")
cell_unzip = find_code_cell("import zipfile")
cell_unet = find_code_cell("class UNet(nn.Module):")
cell_extract_mask = find_code_cell("def extract_mask_from_annotation")
cell_train = find_code_cell("class BoardDataset(Dataset):")
cell_validate = find_code_cell("model.eval()")

new_cells = []

def make_md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def make_code(source_lines):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines}

# 0. Header
header_text = """# 🏒 Hockey Board Segmentation — Pre-train & Fine-tune

This notebook implements the optimal **Scenario B** strategy:
1. **Phase 1 (Pre-training)**: Warps the dataset to a flat, top-down perspective (template space) and trains the U-Net. This allows the network to easily learn board textures and geometry without camera distortion.
2. **Phase 2 (Fine-tuning)**: Loads those pre-trained weights and trains on the original, angled broadcast frames for the final epochs so the network can adapt to real-world camera perspectives.

## Setup steps
1. **Runtime → Change runtime type → T4 GPU** (free tier is enough)
2. Upload the zip from your local machine: `python3 prepare_colab_upload.py` → uploads `colab_training_data.zip`
3. Run all cells
4. Download the output model and replace `src/calibration/board_segmentation_model.pth`"""

new_cells.append(make_md(header_text))

# 1. Upload & Unzip
new_cells.append(make_md("## 1 — Upload & Unzip Data"))
new_cells.append(cell_upload)
new_cells.append(cell_unzip)

# 2. Dependencies
new_cells.append(make_md("## 2 — Install dependencies"))
new_cells.append(make_code(["!pip install -q opencv-python-headless scipy albumentations\n"]))

# 3. Phase 1: Pre-training
new_cells.append(make_md("## 3 — Phase 1: Pre-train on Warped (Canonical) Frames\n\nFirst, we warp the annotated dataset to a canonical template space, and pre-train the model for 50 epochs."))
new_cells.append(make_code([
    "!python scripts/warp_to_template.py \\\n",
    "  --input-dir annotation_frames/new_batch/annotated \\\n",
    "  --mask-dir annotation_frames/new_batch/annotated \\\n",
    "  --output-dir data/warped/train \\\n",
    "  --max-frames 100\n"
]))
new_cells.append(make_code([
    "!python scripts/train_board_segmentation_warped.py \\\n",
    "  --epochs 50 \\\n",
    "  --batch-size 8 \\\n",
    "  --lr 1e-3 \\\n",
    "  --data-dir data/warped\n"
]))

# 4. Phase 2: Fine-tuning Setup
new_cells.append(make_md("## 4 — Phase 2: Fine-tune on Original (Distorted) Frames\n\nNow we load the model we just pre-trained, and fine-tune it directly on the original broadcast frames for 30 epochs."))

# Modify Model definition (UNet) cell to load from warped weights
model_code = cell_unet['source'].copy()
for i, line in enumerate(model_code):
    if "if os.path.exists('board_segmentation_model.pth'):" in line:
        model_code[i] = "if os.path.exists('models/board_segmentation_warped.pth'):\n"
    elif "model.load_state_dict(torch.load('board_segmentation_model.pth', map_location=device))" in line:
        model_code[i] = "    model.load_state_dict(torch.load('models/board_segmentation_warped.pth', map_location=device))\n"
cell_unet['source'] = model_code
new_cells.append(cell_unet)

# Modify Mask Extraction cell (ANN_DIR and board ratio filter)
mask_code = cell_extract_mask['source'].copy()
new_mask_code = []
for line in mask_code:
    if line.startswith("ANN_DIR ="):
        new_mask_code.append("ANN_DIR = 'annotation_frames/new_batch/annotated'\n")
    elif 'pairs.append((orig, mask))' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_mask_code.append(f'{indent}board_ratio = (mask > 0).mean()\n')
        new_mask_code.append(f'{indent}if board_ratio > 0.99:\n')
        new_mask_code.append(f'{indent}    print(f"  SKIP {{base.split(\'/\')[-1]}}: {{board_ratio:.1%}} board (too much)")\n')
        new_mask_code.append(f'{indent}    continue\n')
        new_mask_code.append(line)
    else:
        new_mask_code.append(line)
cell_extract_mask['source'] = new_mask_code
new_cells.append(cell_extract_mask)

# Modify Training loop (EPOCHS)
train_code = cell_train['source'].copy()
for i, line in enumerate(train_code):
    if "EPOCHS         = 150" in line:
        train_code[i] = "EPOCHS         = 30\n"
cell_train['source'] = train_code
new_cells.append(cell_train)

# Validate predictions
new_cells.append(make_md("## 5 — Validate Fine-Tuned Predictions"))
new_cells.append(cell_validate)

# Download
new_cells.append(make_md("## 6 — Download the final model"))
new_cells.append(make_code([
    "from google.colab import files\n",
    "import os\n",
    "if os.path.exists('board_segmentation_model.pth'):\n",
    "    files.download('board_segmentation_model.pth')\n",
    "    print('Downloaded final fine-tuned model!')\n",
    "else:\n",
    "    print('Model not found.')\n"
]))

nb['cells'] = new_cells

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("Successfully rewrote notebook for Scenario B cleanly.")
