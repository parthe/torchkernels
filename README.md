# Testing branch policy

The `testing` branch must always contain the latest `main` and remain ahead
of it with the test files. Keep new tests on `testing`, not on `main`.
After every update to `main`, merge `main` into `testing` and run the tests;
do not merge testing-only commits back into `main`.

Run the matrix-storage tests with:

```bash
python -m unittest torchkernels.utils.data_test -v
```

# Kernel methods in PyTorch
Fast implementations of standard utilities for kernel machines

## Installation
```
pip install -I git+https://github.com/parthe/torchkernels
```
Requires a PyTorch installation

## Stable behavior
Currently this code has been tested with n=10,000 samples.\
with `Python 3.11` and `PyTorch 2.4`

## Test installation with Laplacian kernel
```python
import torch
from torchkernels.kernels.radial import laplacian, LaplacianKernel

n, p, d = 300, 200, 100 # number of samples, centers, dimensions

DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    

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

n, p, d, c = 300, 200, 100, 3 # number of samples, centers, dimensions, outputs

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
Save a kernel matrix from any device without changing its precision:

```python
from torchkernels.utils.data import save_kmat, load_kmat

save_kmat(K, "full_kernel.npz")  # triangle=None, compress=False: store as-is
save_kmat(K, "kernel.npz", triangle="upper", compress=True)
K = load_kmat("kernel.npz")  # CUDA if available, then MPS, then CPU
K_cpu = load_kmat("kernel.npz", device=torch.device("cpu"))  # explicit override
```

The default `triangle=None` preserves the full matrix, including rectangular
and asymmetric matrices. Set `triangle="upper"` or `triangle="lower"` for
approximately half the dense storage of a symmetric square matrix. The diagonal is
included, so the archive holds `n*(n+1)/2` values. The chosen triangle is
authoritative when triangular storage is selected: loading mirrors it to reconstruct a symmetric matrix (without
complex conjugation). Symmetry is assumed rather than checked. Saving uses a
packed CPU buffer and no full GPU copy or triangular index arrays. Loading
needs that packed CPU buffer plus the full reconstructed matrix. Dtype is
preserved, including `bfloat16`; gradients are not saved. `compress=False`
(the default) uses uncompressed NPZ; `True` uses `numpy.savez_compressed`.
Paths are used exactly as supplied, without appending an extension.
Loading chooses CUDA, then MPS, then CPU based on availability at each call.
Pass a `torch.device` to override this choice. The selected device must support
the saved dtype; use CPU explicitly for dtypes unsupported by your accelerator.

- extracting top eigenvectors of a kernel matrix
- Random feature maps for: 
  - Gaussian kernel
  - Laplacian kernel
  - Matern kernel
  - Exponential-power kernel $K(x,z) = \exp(-\|x-z\|^\gamma)$
- Differentiable models
