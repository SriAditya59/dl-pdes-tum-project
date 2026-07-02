import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Using device: {device}')

# ------------------------------
# Load and display dataset
# ------------------------------
with h5py.File('data/ProblemC_dataset.h5', 'r') as f:
    print(f.keys())
    a_train = torch.tensor(np.array(f['a_train']), dtype=dtype)
    u_train = torch.tensor(np.array(f['u_train']), dtype=dtype)
    a_test = torch.tensor(np.array(f['a_test']), dtype=dtype)
    u_test = torch.tensor(np.array(f['u_test']), dtype=dtype)
    X = torch.tensor(np.array(f['X']), dtype=dtype)
    Y = torch.tensor(np.array(f['Y']), dtype=dtype)

print('The shape of X:', X.shape, 'The shape of Y:', Y.shape)
print('The shape of a_train:', a_train.shape)
print('The shape of u_train:', u_train.shape)
print('The shape of a_test:', a_test.shape)
print('The shape of u_test:', u_test.shape)

# Move to GPU and add channel dimension (N, 1, H, W)
a_train = a_train.unsqueeze(1).to(device)
u_train = u_train.unsqueeze(1).to(device)
a_test  = a_test.unsqueeze(1).to(device)
u_test  = u_test.unsqueeze(1).to(device)

# ------------------------------
# FNO layers (implemented from scratch)
# ------------------------------
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.shape[-2], x.shape[-1] // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Low-frequency modes
        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2
        )

        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1=12, modes2=12, width=32):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        # Lift: 1 -> width
        self.fc0 = nn.Linear(1, width)
        # 4 Fourier layers + local bypass
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.w0 = nn.Conv2d(width, width, 1)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.w1 = nn.Conv2d(width, width, 1)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.w2 = nn.Conv2d(width, width, 1)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.w3 = nn.Conv2d(width, width, 1)
        # Project: width -> 1
        self.fc1 = nn.Linear(width, 1)

    def forward(self, x):
        # x: (batch, 1, H, W)
        batch, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1)          # (batch, H, W, 1)
        x = self.fc0(x)                     # (batch, H, W, width)
        x = x.permute(0, 3, 1, 2)           # (batch, width, H, W)

        x1 = self.conv0(x) + self.w0(x)
        x1 = torch.nn.functional.gelu(x1)
        x2 = self.conv1(x1) + self.w1(x1)
        x2 = torch.nn.functional.gelu(x2)
        x3 = self.conv2(x2) + self.w2(x2)
        x3 = torch.nn.functional.gelu(x3)
        x4 = self.conv3(x3) + self.w3(x3)
        x4 = torch.nn.functional.gelu(x4)

        x4 = x4.permute(0, 2, 3, 1)         # (batch, H, W, width)
        x4 = self.fc1(x4)                    # (batch, H, W, 1)
        x4 = x4.permute(0, 3, 1, 2)          # (batch, 1, H, W)
        return x4


# Instantiate model
model = FNO2d(modes1=12, modes2=12, width=32).to(device)
print(f'FNO parameter count: {sum(p.numel() for p in model.parameters())}')

# ------------------------------
# Training
# ------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=20, factor=0.5
)
loss_fn = nn.MSELoss()

batch_size = 16
n_epochs = 300
error_history = []

print("Training FNO ...")
for epoch in range(n_epochs):
    model.train()
    # Shuffle training data each epoch
    perm = torch.randperm(a_train.shape[0])
    a_shuf = a_train[perm]
    u_shuf = u_train[perm]

    epoch_loss = 0.0
    for i in range(0, a_train.shape[0], batch_size):
        a_batch = a_shuf[i:i+batch_size]
        u_batch = u_shuf[i:i+batch_size]

        u_pred = model(a_batch)
        loss = loss_fn(u_pred, u_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (a_train.shape[0] // batch_size)
    scheduler.step(avg_loss)

    if (epoch + 1) % 30 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            u_pred = model(a_test)
            # Average relative L2 error over test samples
            err = torch.mean(
                torch.sqrt(
                    torch.sum(
                        (u_pred - u_test).reshape(u_test.shape[0], -1) ** 2, dim=1
                    ) / torch.sum(u_test.reshape(u_test.shape[0], -1) ** 2, dim=1)
                )
            ).item()
            error_history.append(err)
        print(f"Epoch {epoch+1:3d}: train loss={avg_loss:.6e}, test L2 error={err:.6f}")

# ------------------------------
# Final error
# ------------------------------
model.eval()
with torch.no_grad():
    u_pred = model(a_test)
    final_err = torch.mean(
        torch.sqrt(
            torch.sum((u_pred - u_test).reshape(u_test.shape[0], -1) ** 2, dim=1) /
            torch.sum(u_test.reshape(u_test.shape[0], -1) ** 2, dim=1)
        )
    ).item()
print(f"\nFinal average L2 relative error: {final_err:.6f}")

# ------------------------------
# Visualizations for first test sample
# ------------------------------
a_plot = a_test[0, 0].cpu().numpy()
u_true_plot = u_test[0, 0].cpu().numpy()
u_pred_plot = u_pred[0, 0].cpu().numpy()
abs_err_plot = np.abs(u_pred_plot - u_true_plot)

plt.figure()
plt.imshow(a_plot, origin='lower')
plt.colorbar()
plt.title('Input conductivity a(x,y) (first test sample)')
plt.savefig('project_C/a_input.png')
plt.show()

plt.figure()
plt.imshow(u_pred_plot, origin='lower')
plt.colorbar()
plt.title('Predicted temperature u_pred')
plt.savefig('project_C/u_pred.png')
plt.show()

plt.figure()
plt.imshow(u_true_plot, origin='lower')
plt.colorbar()
plt.title('Ground truth temperature u_true')
plt.savefig('project_C/u_true.png')
plt.show()

plt.figure()
plt.imshow(abs_err_plot, origin='lower')
plt.colorbar()
plt.title('Pointwise absolute error')
plt.savefig('project_C/u_error.png')
plt.show()

# Error vs epoch
# Recorded epochs: 1, 30, 60, ..., 300 (if epoch 1 is recorded, else just 30,60,...)
recorded_epochs = []
if len(error_history) > 0:
    if len(error_history) == (n_epochs // 30) + 1:  # includes epoch 0 (epoch 1)
        recorded_epochs = [1] + list(range(30, n_epochs+1, 30))
    else:
        recorded_epochs = list(range(30, n_epochs+1, 30))
recorded_epochs = recorded_epochs[:len(error_history)]

plt.figure()
plt.plot(recorded_epochs, error_history)
plt.xlabel('Epoch')
plt.ylabel('Average relative L2 error')
plt.yscale('log')
plt.grid(True)
plt.title('Error vs. Epoch')
plt.savefig('project_C/error_vs_epoch.png')
plt.show()