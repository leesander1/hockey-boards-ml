import json
import os

notebook_path = "train_board_segmentation_colab-new.ipynb"

# Read the notebook
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The validation cell is Cell 8 (index 7 in 0-indexed list usually, but let's find it by identifying some code inside it)
validation_cell_idx = None
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and any("fig, axes = plt.subplots(len(pairs)" in line for line in cell["source"]):
        validation_cell_idx = idx
        break

if validation_cell_idx is None:
    # Try finding by threshold = 0.18
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and any("threshold = 0.18" in line for line in cell["source"]):
            validation_cell_idx = idx
            break

if validation_cell_idx is None:
    raise ValueError("Could not find the validation cell in the notebook.")

print(f"Found validation cell at index: {validation_cell_idx}")

# Let's replace the source of this cell with the optimized code
optimized_source = [
    "model.eval()\n",
    "import numpy as np\n",
    "\n",
    "# Compute IoU over ALL pairs without plotting them to avoid crashing Colab\n",
    "print(f\"Calculating IoU over all {len(pairs)} frames...\")\n",
    "ious = []\n",
    "for orig, gt_mask in pairs:\n",
    "    h, w = orig.shape[:2]\n",
    "    # Check if image is already TARGET size or needs resizing\n",
    "    if (w, h) != (TARGET_WIDTH, TARGET_HEIGHT):\n",
    "        img = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),\n",
    "                          (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)\n",
    "    else:\n",
    "        img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "    x = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0\n",
    "    with torch.no_grad():\n",
    "        prob = model(x).squeeze().cpu().numpy()\n",
    "        \n",
    "    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)\n",
    "    pred_mask = (prob > 0.18).astype(np.uint8) * 255\n",
    "\n",
    "    gt_b   = gt_mask   > 0\n",
    "    pred_b = pred_mask > 0\n",
    "    inter  = (gt_b & pred_b).sum()\n",
    "    union  = (gt_b | pred_b).sum()\n",
    "    iou    = inter / union if union > 0 else 0.0\n",
    "    ious.append(iou)\n",
    "\n",
    "mean_iou = np.mean(ious)\n",
    "print(f\"Mean IoU across all validation frames: {mean_iou:.2%}\")\n",
    "\n",
    "# Select a small sample of 6 frames to visualize safely\n",
    "num_vis = min(6, len(pairs))\n",
    "indices = np.linspace(0, len(pairs) - 1, num_vis, dtype=int)\n",
    "print(f\"Visualizing {num_vis} sample frames...\")\n",
    "\n",
    "fig, axes = plt.subplots(num_vis, 3, figsize=(18, 4 * num_vis))\n",
    "if num_vis == 1: axes = [axes]\n",
    "\n",
    "for plot_idx, idx in enumerate(indices):\n",
    "    orig, gt_mask = pairs[idx]\n",
    "    h, w = orig.shape[:2]\n",
    "    \n",
    "    if (w, h) != (TARGET_WIDTH, TARGET_HEIGHT):\n",
    "        img = cv2.resize(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB),\n",
    "                          (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)\n",
    "    else:\n",
    "        img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)\n",
    "        \n",
    "    x = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0\n",
    "    with torch.no_grad():\n",
    "        prob = model(x).squeeze().cpu().numpy()\n",
    "        \n",
    "    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)\n",
    "    pred_mask = (prob > 0.18).astype(np.uint8) * 255\n",
    "    iou = ious[idx]\n",
    "\n",
    "    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "    def overlay(base, mask, color):\n",
    "        vis = base.copy()\n",
    "        vis[mask > 0] = (vis[mask > 0] * 0.4 + np.array(color) * 0.6).astype(np.uint8)\n",
    "        return vis\n",
    "\n",
    "    axes[plot_idx][0].imshow(orig_rgb); axes[plot_idx][0].set_title('Original'); axes[plot_idx][0].axis('off')\n",
    "    axes[plot_idx][1].imshow(overlay(orig_rgb, gt_mask,   [0, 200, 0]))\n",
    "    axes[plot_idx][1].set_title('Ground Truth'); axes[plot_idx][1].axis('off')\n",
    "    axes[plot_idx][2].imshow(overlay(orig_rgb, pred_mask, [0, 120, 255]))\n",
    "    axes[plot_idx][2].set_title(f'Predicted  IoU={iou:.2%}'); axes[plot_idx][2].axis('off')\n",
    "\n",
    "plt.suptitle(f'GT (green) vs Predicted (blue) - Mean IoU: {mean_iou:.2%}', fontsize=14)\n",
    "plt.tight_layout(); plt.show()\n"
]

nb["cells"][validation_cell_idx]["source"] = optimized_source

# Save the notebook back
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Successfully updated validation cell in train_board_segmentation_colab-new.ipynb.")
