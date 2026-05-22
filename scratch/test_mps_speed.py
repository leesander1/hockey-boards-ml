import torch
import torch.nn as nn
import time

print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = torch.device("mps")
x = torch.randn(2, 3, 360, 640, device=device)
y = torch.randn(2, 1, 360, 640, device=device)

class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.final = nn.Conv2d(32, 1, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        e = self.relu(self.conv1(x))
        p = self.pool(e)
        u = self.up(p)
        return self.sigmoid(self.final(u))

model = SimpleUNet().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("Starting test forward/backward pass...")
t0 = time.time()
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()
t1 = time.time()
print(f"Completed test forward/backward in {t1 - t0:.3f} seconds!")
