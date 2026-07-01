import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import h5py

# Choose device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load the dataset
with h5py.File('data/ProblemA_dataset.h5', 'r') as f:
    x_obs = torch.tensor(np.array(f['x_obs']), dtype=torch.float32).to(device)
    u_obs = torch.tensor(np.array(f['u_obs']), dtype=torch.float32).to(device)
    x_test = torch.tensor(np.array(f['x_test']), dtype=torch.float32).to(device)
    k_test = torch.tensor(np.array(f['k_test']), dtype=torch.float32).to(device)
    u_test = torch.tensor(np.array(f['u_test']), dtype=torch.float32).to(device)

# Print the shapes to verify
print('x_obs shape:', x_obs.shape)
print('u_obs shape:', u_obs.shape)