import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt
import h5py
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Using device: {device}')

# ------------------------------
# Load dataset
# ------------------------------
with h5py.File('data/ProblemB_dataset.h5', 'r') as f:
    mu_field = torch.tensor(np.array(f['mu_field']), dtype=dtype).to(device)  # (128,128)
    x_test = torch.tensor(np.array(f['x_test']), dtype=dtype).to(device)       # (16384,2)
    u_test = torch.tensor(np.array(f['u_test']), dtype=dtype).to(device)       # (128,128)

print('x_test:', x_test.shape, 'u_test:', u_test.shape, 'mu_field:', mu_field.shape)

u_test_flat = u_test.reshape(-1, 1)

def fun_mu(x, mu=mu_field, resolution=128):
    delta = 1.0 / (resolution - 1)
    x_loc = torch.floor(x[..., 0] / delta + 0.5).long()
    y_loc = torch.floor(x[..., 1] / delta + 0.5).long()
    loc = y_loc * resolution + x_loc
    mu_flat = mu.reshape(1, -1).to(x.device)
    return mu_flat[0, loc].unsqueeze(-1)

# ------------------------------
# 2D Gauss-Legendre quadrature (200 points per dim = 40,000 total)
# ------------------------------
N_quad = 200
quad_pts_1d, quad_wts_1d = np.polynomial.legendre.leggauss(N_quad)
quad_pts_1d = 0.5 * (quad_pts_1d + 1)
quad_wts_1d = 0.5 * quad_wts_1d

X_grid, Y_grid = np.meshgrid(quad_pts_1d, quad_pts_1d, indexing='ij')
W_grid = np.outer(quad_wts_1d, quad_wts_1d)

x_int = torch.tensor(np.stack([X_grid.ravel(), Y_grid.ravel()], axis=1), dtype=dtype).to(device)
w_int = torch.tensor(W_grid.ravel(), dtype=dtype).to(device).view(-1, 1)

print(f'Integration points: {x_int.shape[0]}')

# ------------------------------
# Deep MLP (tanh activation)
# ------------------------------
class DeepMLP(nn.Module):
    def __init__(self, layers_list):
        super().__init__()
        layers = []
        in_dim = layers_list[0]
        for out_dim in layers_list[1:-1]:
            layers.append(nn.Linear(in_dim, out_dim, dtype=dtype))
            layers.append(nn.Tanh())
            in_dim = out_dim
        self.hidden = nn.Sequential(*layers)
        self.out = nn.Linear(in_dim, layers_list[-1], dtype=dtype)

    def forward(self, x):
        x = self.hidden(x)
        return self.out(x)

# 8 hidden layers, 128 neurons each
nn_model = DeepMLP([2, 128, 128, 128, 128, 128, 128, 128, 128, 1]).to(device)

# Hard BC enforcement
def distance(x):
    return x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])

def u_net(x):
    lift = 1.0 - x[:, 0:1]
    return lift + distance(x) * nn_model(x)

# ------------------------------
# Energy loss (deterministic quadrature)
# ------------------------------
def energy_loss():
    x_int.requires_grad_(True)
    u = u_net(x_int)
    mu_val = fun_mu(x_int)
    du = grad(u, x_int, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    energy_density = 0.5 * mu_val * (du[:, 0:1]**2 + du[:, 1:2]**2)
    return torch.sum(energy_density * w_int)

# ------------------------------
# Training setup (as in reference)
# ------------------------------
optimizer = torch.optim.Adam(nn_model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)

epochs = 10000
error_history = []
time_history = []

print("Training Deep Ritz (tanh, high‑order quadrature)...")
for epoch in range(epochs):
    t0 = time.time()

    L = energy_loss()

    optimizer.zero_grad()
    L.backward()
    optimizer.step()
    scheduler.step()

    time_history.append(time.time() - t0)

    if (epoch + 1) % 500 == 0:
        with torch.no_grad():
            u_pred = u_net(x_test)
            err = torch.sqrt(torch.sum((u_pred - u_test_flat)**2) / torch.sum(u_test_flat**2)).item()
            error_history.append(err)
        print(f"Epoch {epoch+1:5d}: energy loss={L.item():.6e}, L2 error={err:.6f}")

# ------------------------------
# L-BFGS fine‑tuning
# ------------------------------
print("Fine‑tuning with L-BFGS...")
optimizer_lbfgs = torch.optim.LBFGS(nn_model.parameters(), lr=0.5, max_iter=500,
                                    tolerance_grad=1e-9, line_search_fn='strong_wolfe')
def closure():
    optimizer_lbfgs.zero_grad()
    L = energy_loss()
    L.backward()
    return L
optimizer_lbfgs.step(closure)

# Final error
with torch.no_grad():
    u_pred = u_net(x_test)
    err_final = torch.sqrt(torch.sum((u_pred - u_test_flat)**2) / torch.sum(u_test_flat**2)).item()
print(f"\nFinal L2 relative error: {err_final:.6f}")

# ------------------------------
# Plots
# ------------------------------
u_pred_2d = u_pred.cpu().numpy().reshape(128, 128)
u_true_2d = u_test.cpu().numpy()

plt.figure(); plt.imshow(u_pred_2d, origin='lower', extent=[0,1,0,1]); plt.colorbar()
plt.title(f'Predicted pressure (L2 error = {err_final:.4f})'); plt.xlabel('x'); plt.ylabel('y')
plt.savefig('project_B/u_pred.png'); plt.show()

plt.figure(); plt.imshow(u_true_2d, origin='lower', extent=[0,1,0,1]); plt.colorbar()
plt.title('True pressure field'); plt.xlabel('x'); plt.ylabel('y')
plt.savefig('project_B/u_true.png'); plt.show()

abs_err = np.abs(u_pred_2d - u_true_2d)
plt.figure(); plt.imshow(abs_err, origin='lower', extent=[0,1,0,1]); plt.colorbar()
plt.title('Pointwise absolute error'); plt.xlabel('x'); plt.ylabel('y')
plt.savefig('project_B/u_error.png'); plt.show()

plt.figure(); plt.plot(np.arange(500, epochs+1, 500), error_history)
plt.xlabel('Epoch'); plt.ylabel('Relative L2 error')
plt.yscale('log'); plt.title('Error vs. Epoch'); plt.grid(True)
plt.savefig('project_B/error_vs_epoch.png'); plt.show()

cum_time = np.cumsum(time_history)
eval_times = np.cumsum(time_history)[499::500]   # times at recorded epochs
plt.figure(); plt.semilogy(eval_times, error_history)
plt.xlabel('Time (s)'); plt.ylabel('Relative L2 error')
plt.title('Error vs. Time'); plt.grid(True)
plt.savefig('project_B/error_vs_time.png'); plt.show()