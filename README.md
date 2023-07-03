# Kernel Implementations in PyTorch
Fast implementations of standard kernels such as Gaussian, Laplacian, NTK

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

n1 = 1000 # number of samples
n2 = 1000 # number of samples
d = 100  # dimensions

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    DEV_MEM = torch.cuda.get_device_properties(DEVICE).total_memory//1024**3 - 1 # GPU memory in GB, keeping aside 1GB for safety
else:
    DEVICE = torch.device("cpu")
    DEV_MEM = 8 # RAM available for computing

X, Z = torch.randn(n, d, device=DEV), torch.randn(n, d, device=DEV)

kernel_matrix = laplacian(x, y, bandwidth=1.)
print('Laplacian test complete!')
```
