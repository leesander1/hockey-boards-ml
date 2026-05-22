import cv2
import torch
import numpy as np
import sys
sys.path.append('src')
from calibration.ml_board_detector import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = UNet(in_channels=3, out_channels=1).to(device)
model.load_state_dict(torch.load('src/calibration/board_segmentation_model_unet.pth', map_location=device))
model.eval()

cap = cv2.VideoCapture('data/videos/2026-05-12 21-15-46.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 5)
ret, frame = cap.read()
h, w = frame.shape[:2]

# Try both 320x176 and 640x360
for size in [(320, 176), (640, 360)]:
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_tensor)
        
    pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
    cv2.imwrite(f'scratch/debug/raw_unet_{size[0]}x{size[1]}.png', (pred_np * 255).astype(np.uint8))

