# Kernel methods in PyTorch
Fast implementations of standard utilities 

# Installation
```
pip install -I git+https://github.com/parthe/torchkernels
```
Requires a PyTorch installation

## Stable behavior
Currently this code has been tested with n=10,000 samples.\
with Python 3.9 and `PyTorch >= 1.13`

# Test installation with Laplacian kernel
```python
import torch
from torchkernels.kernels.radial import laplacian, LaplacianKernel

n = 300 # number of samples
p = 200 # number of centers
d = 100  # dimensions

is_cuda = torch.cuda.is_available()
DEV = torch.device("cuda") if is_cuda else torch.device("cpu")    

X = torch.randn(n, d, device=DEV)
Z = torch.randn(p, d, device=DEV)

kernel_matrix1 = laplacian(X, Z, bandwidth=1.)

K = LaplacianKernel(bandwidth=1.)
kernel_matrix2 = K(X, Z)

torch.testing.assert_close(kernel_matrix1, kernel_matrix2, msg='Laplacian test failed')
print('Laplacian test complete!')
```

## Currently supported Kernels
- Laplacian, Gaussian, Dispersal (Exponential power kernel)
- Normalized dot-product kernel for arbitrary functions
- Neural Network Gaussian Process (NNGP) and Tangent Kernel (NTK) with ReLU activations

## Other utilities
- top eigenvectors of kernel matrix
- Random feature maps for the Laplacian, Gaussian, Matern and the ExpPower kernel
