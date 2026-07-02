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
# Load dataset
# ------------------------------
with h5py.File('data/ProblemD_dataset.h5', 'r') as file:
    print(file.keys())
    t_mesh = torch.tensor(np.array(file['t_mesh']), dtype=dtype)          # (200,1)
    x_mesh = torch.tensor(np.array(file['x_mesh']), dtype=dtype)          # (256,1)
    a_test = torch.tensor(np.array(file['a_test']), dtype=dtype)          # (200,256)
    u_test = torch.tensor(np.array(file['u_test']), dtype=dtype)          # (200,200,256)
    a_train_labeled = torch.tensor(np.array(file['a_train_labeled']), dtype=dtype)    # (200,256)
    u_train_labeled = torch.tensor(np.array(file['u_train_labeled']), dtype=dtype)    # (200,200,256)
    # Unlabeled data not used in this supervised version

print('t_mesh:', t_mesh.shape, 'x_mesh:', x_mesh.shape)
print('a_train_labeled:', a_train_labeled.shape, 'u_train_labeled:', u_train_labeled.shape)
print('a_test:', a_test.shape, 'u_test:', u_test.shape)

# Move to GPU
a_train = a_train_labeled.to(device)   # (200,256)
u_train = u_train_labeled.to(device)   # (200,200,256)
a_test = a_test.to(device)
u_test = u_test.to(device)
x_mesh = x_mesh.to(device).squeeze()   # (256,)
t_mesh = t_mesh.to(device).squeeze()   # (200,)

# ------------------------------
# DeepONet (same as reference)
# ------------------------------
class DeepONet(nn.Module):
    def __init__(self, p=128):
        super().__init__()
        # Branch: 256 -> 128,128,128,128 -> p
        self.branch = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, p)
        )
        # Trunk: 2 -> 128,128,128,128 -> p
        self.trunk = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, p)
        )
        self.b0 = nn.Parameter(torch.zeros(1, dtype=dtype))

    def forward(self, a, xt):
        # a: (batch, 256), xt: (batch, N, 2)
        branch_out = self.branch(a)                # (batch, p)
        trunk_out = self.trunk(xt)                 # (batch, N, p)
        return torch.sum(branch_out.unsqueeze(1) * trunk_out, dim=2) + self.b0

model = DeepONet(p=128).to(device)
print(f'Parameters: {sum(p.numel() for p in model.parameters())}')

# ------------------------------
# Loss (L2 norm, as in reference)
# ------------------------------
def loss_fn(u_pred, u_true):
    """Mean of per‑sample relative L2 norms."""
    diff = u_pred.reshape(u_pred.shape[0], -1) - u_true.reshape(u_true.shape[0], -1)
    return torch.norm(diff, dim=1).mean()

# ------------------------------
# Training setup
# ------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

epochs = 2000
batch_size = 32
error_history = []

print("Training supervised DeepONet ...")
for epoch in range(epochs):
    model.train()
    # Shuffle training data
    perm = torch.randperm(a_train.shape[0])
    a_shuf = a_train[perm]
    u_shuf = u_train[perm]

    epoch_loss = 0.0
    batches = 0
    for i in range(0, a_train.shape[0], batch_size):
        a_batch = a_shuf[i:i+batch_size]
        u_batch = u_shuf[i:i+batch_size]
        B = a_batch.shape[0]

        # Sample random spatiotemporal points from the full grid
        N_pts = 2000
        t_idx = torch.randint(0, 200, (N_pts,), device=device)
        x_idx = torch.randint(0, 256, (N_pts,), device=device)
        xt = torch.stack([t_mesh[t_idx], x_mesh[x_idx]], dim=1)   # (N_pts, 2)
        xt = xt.unsqueeze(0).expand(B, -1, -1)                     # (B, N_pts, 2)
        u_target = u_batch[:, t_idx, x_idx]                        # (B, N_pts)

        u_pred = model(a_batch, xt)
        loss = loss_fn(u_pred, u_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        batches += 1

    scheduler.step()

    if (epoch+1) % 200 == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            total_err = 0.0
            for j in range(a_test.shape[0]):
                a_j = a_test[j:j+1]
                u_true_j = u_test[j]
                tg, xg = torch.meshgrid(t_mesh, x_mesh, indexing='ij')
                xt_grid = torch.stack([tg.reshape(-1), xg.reshape(-1)], dim=1).unsqueeze(0)
                u_pred_j = model(a_j, xt_grid).reshape(200, 256)
                total_err += torch.sqrt(torch.sum((u_pred_j - u_true_j)**2) / torch.sum(u_true_j**2)).item()
            avg_err = total_err / a_test.shape[0]
            error_history.append(avg_err)
        print(f"Epoch {epoch+1:4d}: loss={epoch_loss/batches:.6e}, test L2 error={avg_err:.6f}")

# Final error
model.eval()
with torch.no_grad():
    total_err = 0.0
    for j in range(a_test.shape[0]):
        a_j = a_test[j:j+1]
        u_true_j = u_test[j]
        tg, xg = torch.meshgrid(t_mesh, x_mesh, indexing='ij')
        xt_grid = torch.stack([tg.reshape(-1), xg.reshape(-1)], dim=1).unsqueeze(0)
        u_pred_j = model(a_j, xt_grid).reshape(200, 256)
        total_err += torch.sqrt(torch.sum((u_pred_j - u_true_j)**2) / torch.sum(u_true_j**2)).item()
    final_err = total_err / a_test.shape[0]
print(f"\nFinal average L2 relative error: {final_err:.6f}")

# ------------------------------
# Plots (first test instance)
# ------------------------------
inx = 0
with torch.no_grad():
    a_plot = a_test[inx].cpu().numpy()
    u_true_plot = u_test[inx].cpu().numpy()
    tg, xg = torch.meshgrid(t_mesh, x_mesh, indexing='ij')
    xt_grid = torch.stack([tg.reshape(-1), xg.reshape(-1)], dim=1).unsqueeze(0)
    u_pred_plot = model(a_test[inx:inx+1], xt_grid).reshape(200, 256).cpu().numpy()
    abs_err = np.abs(u_pred_plot - u_true_plot)

    plt.figure(); plt.plot(x_mesh.cpu(), a_plot); plt.xlabel('x'); plt.ylabel('u(x,0)')
    plt.title('Initial condition'); plt.savefig('project_D/initial_condition.png'); plt.show()

    plt.figure(); plt.contourf(x_mesh.cpu(), t_mesh.cpu(), u_pred_plot, levels=50, cmap='jet')
    plt.colorbar(); plt.xlabel('x'); plt.ylabel('t'); plt.title('Predicted velocity')
    plt.savefig('project_D/u_pred.png'); plt.show()

    plt.figure(); plt.contourf(x_mesh.cpu(), t_mesh.cpu(), u_true_plot, levels=50, cmap='jet')
    plt.colorbar(); plt.xlabel('x'); plt.ylabel('t'); plt.title('True velocity')
    plt.savefig('project_D/u_true.png'); plt.show()

    plt.figure(); plt.contourf(x_mesh.cpu(), t_mesh.cpu(), abs_err, levels=50, cmap='jet')
    plt.colorbar(); plt.xlabel('x'); plt.ylabel('t'); plt.title('Pointwise absolute error')
    plt.savefig('project_D/u_error.png'); plt.show()

# Error vs. epoch (corrected plot)
recorded_epochs = [1] + list(range(200, epochs+1, 200))   # epoch 1, 200, 400, ..., 2000
recorded_epochs = recorded_epochs[:len(error_history)]       # trim to match
plt.figure()
plt.plot(recorded_epochs, error_history)
plt.xlabel('Epoch'); plt.ylabel('Average relative L2 error')
plt.yscale('log'); plt.grid(True); plt.title('Error vs. Epoch')
plt.savefig('project_D/error_vs_epoch.png'); plt.show()