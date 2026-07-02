import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import h5py
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
print(f'Using device: {device}')

# ------------------------------
# Load dataset
# ------------------------------
with h5py.File('data/ProblemA_dataset.h5', 'r') as f:
    x_obs = torch.tensor(np.array(f['x_obs']), dtype=dtype).reshape(-1,1).to(device)
    u_obs = torch.tensor(np.array(f['u_obs']), dtype=dtype).reshape(-1,1).to(device)
    x_test = torch.tensor(np.array(f['x_test']), dtype=dtype).reshape(-1,1).to(device)
    k_test = torch.tensor(np.array(f['k_test']), dtype=dtype).reshape(-1,1).to(device)
    u_test = torch.tensor(np.array(f['u_test']), dtype=dtype).reshape(-1,1).to(device)

# ------------------------------
# Neural network for u(x)
# ------------------------------
class MLP(nn.Module):
    def __init__(self, layers_list):
        super().__init__()
        layers = []
        in_dim = layers_list[0]
        for out_dim in layers_list[1:]:
            layers.append(nn.Linear(in_dim, out_dim, dtype=dtype))
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.net[:-1]:
            x = layer(x)
            x = torch.sin(np.pi * x + np.pi) + torch.sin(x)
        x = self.net[-1](x)
        return x

u_net = MLP([1, 60, 60, 60, 60, 1]).to(device)

# ------------------------------
# Parametric k(x) – two Gaussians + baseline
# ------------------------------
c0 = torch.tensor(1.0, dtype=dtype, device=device, requires_grad=True)
c1 = torch.tensor(1.0, dtype=dtype, device=device, requires_grad=True)
mu1 = torch.tensor(0.25, dtype=dtype, device=device, requires_grad=True)
sigma1 = torch.tensor(0.1, dtype=dtype, device=device, requires_grad=True)
c2 = torch.tensor(1.0, dtype=dtype, device=device, requires_grad=True)
mu2 = torch.tensor(0.75, dtype=dtype, device=device, requires_grad=True)
sigma2 = torch.tensor(0.1, dtype=dtype, device=device, requires_grad=True)

def k_parametric(x):
    g1 = torch.exp(-((x - mu1) ** 2) / (2 * sigma1**2 + 1e-6))
    g2 = torch.exp(-((x - mu2) ** 2) / (2 * sigma2**2 + 1e-6))
    return c0 + c1 * g1 + c2 * g2

# ------------------------------
# Collocation points & DataLoader
# ------------------------------
n_collocation = 10000
x_in = torch.rand(n_collocation, 1, device=device)

class MyDataset(Dataset):
    def __init__(self, x): self.x = x
    def __getitem__(self, idx): return self.x[idx]
    def __len__(self): return len(self.x)

dataloader = DataLoader(MyDataset(x_in), batch_size=500, shuffle=True)

x_bd = torch.tensor([[0.0], [1.0]], dtype=dtype).to(device)
u_bd = torch.zeros(2, 1, dtype=dtype).to(device)

# Fixed validation collocation points (used for early stopping, NOT x_test)
x_val = torch.rand(2000, 1, device=device)

# ------------------------------
# Loss functions
# ------------------------------
mse = nn.MSELoss()
f_val = 9.81

def loss_data():
    return mse(u_net(x_obs), u_obs)

def loss_bc():
    return mse(u_net(x_bd), u_bd)

def loss_pde(x):
    x.requires_grad_(True)
    u = u_net(x)
    k = k_parametric(x)
    du = grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    flux = k * du
    dflux = grad(flux, x, grad_outputs=torch.ones_like(flux), create_graph=True)[0]
    residual = -dflux - f_val
    return torch.mean(residual ** 2)

# ------------------------------
# Training setup
# ------------------------------
k_params = [c0, c1, mu1, sigma1, c2, mu2, sigma2]
params = list(u_net.parameters()) + k_params
optimizer = torch.optim.Adam(params, lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.95)   # very slow decay

w_data = 10.0
w_pde  = 2.0
w_bc   = 10.0

epochs = 600
error_u_hist, error_k_hist = [], []
time_per_epoch = []
val_pde_loss_best = float('inf')
best_state = None

print("Training PINN with two Gaussians + early stopping (corrected)...")
for epoch in range(epochs):
    t0 = time.time()
    for x_batch in dataloader:
        L_data = loss_data()
        L_pde  = loss_pde(x_batch.view(-1, 1))
        L_bc   = loss_bc()
        total_loss = w_data * L_data + w_pde * L_pde + w_bc * L_bc

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

    scheduler.step()
    time_per_epoch.append(time.time() - t0)

    # ---- Validation PDE loss (no gradient needed, but autograd must be active) ----
    # We compute the loss using the current model parameters, then discard the graph.
    val_pde = loss_pde(x_val).item()          # <-- no 'with torch.no_grad()'
    if val_pde < val_pde_loss_best:
        val_pde_loss_best = val_pde
        best_state = {
            'u_net': {k: v.cpu().clone() for k, v in u_net.state_dict().items()},
            'k_params': [p.detach().cpu().clone() for p in k_params]
        }

    # ---- Test errors (evaluation only) ----
    with torch.no_grad():
        u_pred = u_net(x_test)
        k_pred = k_parametric(x_test)
        err_u = torch.sqrt(torch.sum((u_pred - u_test)**2) / torch.sum(u_test**2)).item()
        err_k = torch.sqrt(torch.sum((k_pred - k_test)**2) / torch.sum(k_test**2)).item()
        error_u_hist.append(err_u)
        error_k_hist.append(err_k)

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1:3d}: loss={total_loss.item():.6e}, val_pde={val_pde:.6e}, err_u={err_u:.4f}, err_k={err_k:.4f}")

# Restore best model
if best_state is not None:
    u_net.load_state_dict(best_state['u_net'])
    for p, new_val in zip(k_params, best_state['k_params']):
        p.data.copy_(new_val.to(device))
    print("Restored best model (lowest validation PDE loss).")

# ------------------------------
# Fine‑tune with L‑BFGS (short polish)
# ------------------------------
print("Fine‑tuning with L‑BFGS...")
optimizer_lbfgs = torch.optim.LBFGS(params, lr=0.5, max_iter=200,
                                    tolerance_grad=1e-9, line_search_fn='strong_wolfe')
def closure():
    optimizer_lbfgs.zero_grad()
    x_c = torch.rand(2000, 1, device=device)
    L_data = loss_data()
    L_pde  = loss_pde(x_c)
    L_bc   = loss_bc()
    total = w_data * L_data + w_pde * L_pde + w_bc * L_bc
    total.backward()
    return total
optimizer_lbfgs.step(closure)

# Final errors
with torch.no_grad():
    u_pred = u_net(x_test)
    k_pred = k_parametric(x_test)
    err_u_final = torch.sqrt(torch.sum((u_pred - u_test)**2) / torch.sum(u_test**2)).item()
    err_k_final = torch.sqrt(torch.sum((k_pred - k_test)**2) / torch.sum(k_test**2)).item()

print(f"\nFinal L2 errors — u: {err_u_final:.4f}, k: {err_k_final:.4f}")
print(f"Learned k parameters: c0={c0.item():.3f}, c1={c1.item():.3f}, mu1={mu1.item():.3f}, sigma1={sigma1.item():.3f}, "
      f"c2={c2.item():.3f}, mu2={mu2.item():.3f}, sigma2={sigma2.item():.3f}")

# ------------------------------
# Plots
# ------------------------------
x_t = x_test.cpu().numpy().flatten()
u_t = u_test.cpu().numpy().flatten()
k_t = k_test.cpu().numpy().flatten()
u_p = u_pred.cpu().numpy().flatten()
k_p = k_pred.cpu().numpy().flatten()

plt.figure()
plt.plot(x_t, k_t, 'k-', label='True k(x)')
plt.plot(x_t, k_p, 'r--', label='Predicted k(x)')
plt.xlabel('x'); plt.ylabel('k(x)')
plt.legend()
plt.title(f'Young\'s modulus recovery (L2 error = {err_k_final:.4f})')
plt.savefig('project_A/k_comparison.png')
plt.show()

plt.figure()
plt.plot(x_t, u_t, 'k-', label='True u(x)')
plt.plot(x_t, u_p, 'b--', label='Predicted u(x)')
plt.scatter(x_obs.cpu(), u_obs.cpu(), s=5, c='red', alpha=0.5, label='Noisy observations')
plt.xlabel('x'); plt.ylabel('u(x)')
plt.legend()
plt.title(f'Displacement field (L2 error = {err_u_final:.4f})')
plt.savefig('project_A/u_comparison.png')
plt.show()

plt.figure()
plt.plot(x_t, np.abs(k_p - k_t), 'm-')
plt.xlabel('x'); plt.ylabel('|k_pred - k_true|')
plt.title('Pointwise absolute error for k(x)')
plt.savefig('project_A/k_pointwise_error.png')
plt.show()

plt.figure()
plt.plot(error_k_hist, 'r-', label='k error')
plt.plot(error_u_hist, 'b-', label='u error')
plt.yscale('log')
plt.xlabel('Epoch'); plt.ylabel('Relative L2 error')
plt.legend()
plt.title('Error vs. Epoch')
plt.grid(True)
plt.savefig('project_A/error_vs_epoch.png')
plt.show()

cum_time = np.cumsum(time_per_epoch)
plt.figure()
plt.semilogy(cum_time, error_k_hist, 'r-', label='k error')
plt.semilogy(cum_time, error_u_hist, 'b-', label='u error')
plt.xlabel('Time (s)'); plt.ylabel('Relative L2 error')
plt.legend()
plt.title('Error vs. Time')
plt.grid(True)
plt.savefig('project_A/error_vs_time.png')
plt.show()