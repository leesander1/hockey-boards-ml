import json

file_path = 'train_board_segmentation_colab.ipynb'
with open(file_path, 'r') as f:
    notebook = json.load(f)

new_aug_code = [
    "    @staticmethod\n",
    "    def _aug(img, mask):\n",
    "        h, w = img.shape[:2]\n",
    "        # Horizontal flip\n",
    "        if np.random.rand() > 0.5:\n",
    "            img = cv2.flip(img, 1); mask = cv2.flip(mask, 1)\n",
    "        # Aggressive Brightness/contrast\n",
    "        a = np.random.uniform(0.60, 1.40); b = int(np.random.uniform(-40, 40))\n",
    "        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)\n",
    "        # HSV jitter\n",
    "        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)\n",
    "        hsv[...,0] = (hsv[...,0] + np.random.randint(-15, 15)) % 180\n",
    "        hsv[...,1] = np.clip(hsv[...,1] + np.random.randint(-35, 35), 0, 255)\n",
    "        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)\n",
    "        # Random crop & scale\n",
    "        s = np.random.uniform(0.75, 1.0)\n",
    "        nh, nw = int(h*s), int(w*s)\n",
    "        y0 = np.random.randint(0, h-nh+1); x0 = np.random.randint(0, w-nw+1)\n",
    "        img  = cv2.resize(img[y0:y0+nh, x0:x0+nw], (w, h))\n",
    "        mask = cv2.resize(mask[y0:y0+nh, x0:x0+nw], (w, h), interpolation=cv2.INTER_NEAREST)\n",
    "        # Add random Gaussian noise\n",
    "        if np.random.rand() > 0.5:\n",
    "            noise = np.random.normal(0, np.random.uniform(10, 25), img.shape)\n",
    "            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)\n",
    "        # Gaussian blur (simulates camera defocus)\n",
    "        if np.random.rand() > 0.5:\n",
    "            k = np.random.choice([3, 5, 7])\n",
    "            img = cv2.GaussianBlur(img, (k, k), 0)\n",
    "        # Random Cutout (simulate players blocking the boards)\n",
    "        if np.random.rand() > 0.3:\n",
    "            for _ in range(np.random.randint(1, 4)):\n",
    "                cy, cx = np.random.randint(0, h), np.random.randint(0, w)\n",
    "                ch, cw = np.random.randint(20, 80), np.random.randint(20, 80)\n",
    "                y1, y2 = max(0, cy - ch//2), min(h, cy + ch//2)\n",
    "                x1, x2 = max(0, cx - cw//2), min(w, cx + cw//2)\n",
    "                img[y1:y2, x1:x2] = np.random.randint(0, 255, (3,))\n",
    "                mask[y1:y2, x1:x2] = 0\n",
    "        return img, mask\n"
]

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        # Find the start and end of _aug function
        start_idx = -1
        end_idx = -1
        for i, line in enumerate(source):
            if "def _aug(img, mask):" in line:
                start_idx = i - 1 # include @staticmethod
            if start_idx != -1 and "return img, mask" in line:
                end_idx = i
                break
        
        if start_idx != -1 and end_idx != -1:
            # Replace the lines
            cell['source'] = source[:start_idx] + new_aug_code + source[end_idx+1:]
            break

with open(file_path, 'w') as f:
    json.dump(notebook, f, indent=1)
