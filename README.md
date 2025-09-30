# Kernel methods in PyTorch
Fast implementations of standard utilities for kernel machines

## Installation
```
pip install -I git+https://github.com/parthe/torchkernels
```
Requires a PyTorch installation

## Stable behavior
Currently this code has been tested with n=10,000 samples.\
with Python 3.9 and `PyTorch >= 1.13`

## Test installation with Laplacian kernel
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

kernel_matrix1 = laplacian(X, Z, length_scale=1.)

K = LaplacianKernel(length_scale=1.)
kernel_matrix2 = K(X, Z)

torch.testing.assert_close(kernel_matrix1, kernel_matrix2, msg='Laplacian test failed')
print('Laplacian test complete!')
```

## Example: Differentiating the kernel function
```python
import torch
from torchkernels.kernels.radial import laplacian
from torch.func import vmap, grad, jacrev

n, p, d, c = 300, 200, 100, 3

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    

X = torch.randn(n, d, device=DEV)
Z = torch.randn(p, d, device=DEV)
a = torch.randn(p, device=DEV)
A = torch.randn(p, c, device=DEV)

f = lambda x: laplacian(x, Z, length_scale=1., in_place=False).squeeze() @ a
print(vmap(grad(f))(X).shape)
# torch.Size([300, 100])

F = lambda x: laplacian(x, Z, length_scale=1., in_place=False).squeeze() @ A
print(vmap(jacrev(F))(X).shape)
# torch.Size([300, 3, 100])
```

# Random features
See an example of [Logistic regression with random features of the Laplacian kernel](https://github.com/parthe/torchkernels/blob/main/demos/feature_maps/logistic_regression.ipynb).

## Currently supported Kernels
- Laplacian, Gaussian, Dispersal (Exponential power kernel)
- Normalized dot-product kernel for arbitrary functions
- Neural Network Gaussian Process (NNGP) and Tangent Kernel (NTK) with ReLU activations

## Other utilities
- extracting top eigenvectors of a kernel matrix
- Random feature maps for: 
  - Gaussian kernel
  - Laplacian kernel
  - Matern kernel
  - Exponential-power kernel $K(x,z) = \exp(-\|x-z\|^\gamma)$
- Differentiable models
