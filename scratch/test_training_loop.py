import torch
import torch.nn as nn
import torch.optim as optim
import time

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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))
        d3 = self.dec3(torch.cat([self.upconv3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))
        return self.sigmoid(self.final(d1))

device = torch.device("mps")
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCELoss()

print("Running 10 batches on MPS...")
t0 = time.time()
for i in range(10):
    img = torch.randn(16, 3, 360, 640, device=device)
    mask = torch.randn(16, 1, 360, 640, device=device)
    
    optimizer.zero_grad()
    pred = model(img)
    bce = criterion(pred, mask)
    intersection = (pred * mask).sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1.0) / (pred.sum(dim=(1, 2, 3)) + mask.sum(dim=(1, 2, 3)) + 1.0))
    loss = bce + dice.mean()
    loss.backward()
    optimizer.step()
    print(f"Batch {i+1} done")
t1 = time.time()
print(f"Completed 10 batches in {t1 - t0:.3f} seconds!")
