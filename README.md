# Kernel utilities in PyTorch
Fast implementations of standard kernels such as Gaussian, Laplacian, NTK, and utilities for kernel methods

# Installation
```
pip install git+https://github.com/parthe/pytorch_kernel_implementations
```
Requires a PyTorch installation

## Stable behavior
Currently this code has been tested with n=10,000 samples.\
with Python 3.9 and `PyTorch >= 1.13`

# Test installation with Laplacian kernel
```python
import torch
from kernels import laplacian

n = 300 # number of samples
p = 200 # number of centers
d = 100  # dimensions

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3 - 1 # GPU memory in GB, keeping aside 1GB for safety
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8 # RAM available for computing

X = torch.randn(n, d, device=DEVICE)
Z = torch.randn(p, d, device=DEVICE)

kernel_matrix = laplacian(X, Z, bandwidth=1.)
print('Laplacian test complete!')
```
