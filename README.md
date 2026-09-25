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

## Resident multi-GPU matrix-vector multiplication

```python
from torchkernels.utils import KernelLinearOperator, MultiGpuKernelLinearOperator

# K: dense symmetric tensor; x: matching-dtype vector of shape (n,).
Kmat = KernelLinearOperator(K)  # automatically use multiple visible CUDA GPUs
y = Kmat @ x  # fresh CPU vector; later products reuse GPU storage and buffers
```

`KernelLinearOperator(K)` automatically returns `MultiGpuKernelLinearOperator`
when more than one CUDA GPU is visible to the process. Selection honors
`CUDA_VISIBLE_DEVICES`; explicit `devices=[...]` overrides which visible GPUs
participate. With `devices=[0]` or only one visible CUDA GPU, it uses a packed
single-GPU snapshot. With no CUDA GPUs it uses a packed snapshot on the source
tensor's CPU or MPS device. MPS has no multi-device dispatch.

`KernelLinearOperator` inherits from `SymmetricLinearOperator`, defined in
`torchkernels.utils.packed` and also exported from `torchkernels.utils`.
All paths read the upper triangle, preserve symmetry without conjugation,
return CPU vectors, and retain exactly `n*(n+1)/2` matrix values. Single-device
CPU/MPS multiplication uses packed triangle views; CUDA uses cuBLAS SYMV.
No backend reconstructs the full matrix for multiplication.

```python
from torchkernels.utils.packed import SymmetricLinearOperator

S = SymmetricLinearOperator(K, device="cuda:0")  # explicitly one packed device
Kmat = KernelLinearOperator(K)                  # automatically choose CUDA GPUs
y = Kmat @ x
assert isinstance(Kmat, SymmetricLinearOperator)
```

Balancing and calibration apply to the multi-GPU path only; CUDA memory
headroom settings apply to both single- and multi-GPU paths. Memory remains the default. Explicit construction of
`MultiGpuKernelLinearOperator` is still supported, including on one CUDA GPU.
Subclasses of `KernelLinearOperator` construct normally without redispatch.

`MultiGpuKernelLinearOperator` specializes `KernelLinearOperator` (available
from `torchkernels.linalg` or `torchkernels.utils`). `Kmat(x)` and
`Kmat.matvec(x)` also work. This is a vector interface, not a Tensor subclass;
`torch.matmul(Kmat, x)` and matrix right-hand sides are not supported.

### Paired diagonal storage and cuBLAS

All diagonal blocks reside on the **first selected GPU** (`cuda:0` by default).
Equal-size diagonal blocks are paired in a column-major `(b+1, b)` buffer:

```text
A00 A01 A02
B00 A11 A12
B10 B11 A22
B20 B21 B22
```

cuBLAS SYMV reads A's upper triangle at offset zero, then B's lower triangle
at offset one element; both use `lda=b+1`. B is not reversed, and vectors need
no permutation. The two calls compute the full symmetric block products
without expanding either block into a dense scratch matrix. For odd n, a
final 1-by-1 diagonal block stores the remaining scalar. All other diagonal
blocks are paired without padding. Total matrix storage is exactly
`n*(n+1)/2` values, excluding vector buffers and library workspaces.

Only the input's upper triangle is read. Lower-triangle entries in paired
storage are populated from the source upper triangle by symmetry. Complex
matrices are symmetric, not Hermitian: there is no conjugation. Supported
dtypes are `float32`, `float64`, `complex64`, and `complex128`; cuBLAS SYMV has
no half/bfloat16 entry point, so these dtypes are rejected explicitly.

All off-diagonal tiles are rectangles. A tile `B = K[I, J]` contributes
`B @ x[J]` to output rows I and `B.T @ x[I]` to output rows J. Rectangles can
reside on any selected GPU, including the first one.

### Memory versus compute balancing

```python
# Proportional to currently usable GPU memory (the default).
Kmat = MultiGpuKernelLinearOperator(K, devices=[0, 1], load_balance="memory")

# Automatically measure effective dtype-matched GEMV TFLOPS.
Kmat = MultiGpuKernelLinearOperator(K, devices=[0, 1, 2], load_balance="compute")
print(Kmat.tflops, Kmat.tflops_source)  # selected-device order, "benchmark"
print(Kmat.devices, Kmat.storage_numel, Kmat.estimated_flops)

# Optional explicit ratings bypass calibration (example numbers only).
Kmat = MultiGpuKernelLinearOperator(
    K, devices=[0, 1, 2], load_balance="compute", tflops=[10.0, 20.0, 30.0],
)
```

`memory` mode balances total stored matrix elements in proportion to usable
capacity, including the mandatory diagonal allocation on the first GPU.
`compute` mode automatically benchmarks both directions of a rectangular
matrix-vector product using the matrix dtype and CUDA events. Three warmup
pairs precede three timed samples of ten pairs; the median gives effective
TFLOPS. Calibration uses a common tile width on all eligible devices, bounded
by `block_size`, half the matrix size (minimum one), 1024, and available memory.
At most 1024² matrix values plus two short vectors are allocated at a time.
Measurements include the effects of cache and current GPU activity. They are
not theoretical peak hardware ratings and do not time SYMV or inter-GPU/CPU
transfers. Calibration repeats for each newly constructed compute-mode operator.
Memory mode remains the default and performs no benchmark.

Optionally supply `tflops=[...]` to skip calibration. Provide one positive finite
rating per selected GPU, in exactly the `devices` order. Ratings should use the
matrix's precision and ordinary arithmetic rather than tensor-core/TF32 peaks.
`tflops_source` reports `"benchmark"`, `"provided"`, or `None` (memory mode).
This mode aims to equalize estimated work divided by its rating, subject to
hard memory caps. Free memory is queried again after calibration to account for
CUDA workspaces and allocator caches. Zero-capacity GPUs are not benchmarked;
their automatic rating is zero and they remain excluded from placement.
Passing `tflops` in memory mode raises an error rather than silently ignoring it.

Planning counts a symmetric b-by-b diagonal block as approximately `2*b*b`
real operations and a rectangular tile as four operations per stored element.
Complex products use four times these real-operation estimates. Mandatory
diagonal work is included before distributing rectangles. Diagonal blocks
shrink when necessary to respect the first GPU's fair share or capacity;
`block_size=1024` is an upper bound, not a fixed tile width. With three equal
GPUs, for example, planning can choose four diagonal blocks instead of forcing
half the matrix onto the first GPU. Even at minimum block width, that GPU must
still hold and process the n principal diagonal values.

Work estimates omit bandwidth, transfer cost, kernel-launch overhead, and
library-dependent performance. TFLOPS-based balancing is a placement policy,
not a guarantee of equal measured runtimes. Placement remains fixed until the
operator is rebuilt. Very small tiles increase launch and planning overhead.

### Memory budget, introspection, and runtime

By default, `memory_fraction=0.9` allows 90% of currently free memory, less
`reserve_bytes=64 * 1024**2` per GPU and two full vectors (input and partial
output). There is no diagonal reconstruction buffer. `reserve_bytes` leaves
headroom for CUDA/cuBLAS workspaces. Contexts and the private SYMV handle are
created before free memory is queried. Fragmentation and concurrent external
allocations can still cause CUDA OOM. Insufficient aggregate capacity or a
first GPU unable to fit the principal diagonal raises `MemoryError` before
allocating matrix storage. GPUs assigned no matrix data are skipped.

`diagonal_blocks` lists global `(start, stop)` ranges, all on `devices[0]`.
`tile_ranges` contains only rectangular `(row_start, row_stop, column_start,
column_stop)` tiles per active GPU, with exclusive stops. `storage_numel`
includes diagonal storage on the first GPU. `estimated_flops` lists approximate
real arithmetic operations per active GPU. `tflops` retains the supplied or measured ratings
in the original selected-device order, including any subsequently skipped GPU.

CPU matrix input is recommended: a CUDA source remains caller-owned and uses
additional GPU memory. The operator retains no source reference or autograd
graph. CUDA vectors are staged through one CPU copy before broadcast, and cross-device
source matrix tiles use bounded CPU staging. This avoids reliance on peer-copy
ordering/support. Each active GPU receives the complete input vector, computes on a
private stream, and returns a partial vector summed on CPU. Calls are serialized
for safe buffer reuse. A single CUDA GPU works; no GPU raises `RuntimeError`.

The SYMV binding loads the cuBLAS library supplied with CUDA-enabled PyTorch
lazily, owns a private cuBLAS handle, and uses PyTorch tensor pointers and CUDA
streams. It requires no CuPy dependency or runtime compilation. See NVIDIA's
[SYMV documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-symv).

On the `testing` branch:

```bash
python -m unittest torchkernels.utils.packed_test torchkernels.linalg.linear_operator_test torchkernels.utils.multigpu_matvec_test -v
```

CPU tests cover planning, exact packing, numeric products, and cuBLAS pointer
arguments through an emulated API. CUDA tests exercise the real library on
one, two, and three GPUs when available. CPU emulation does not establish
CUDA correctness or performance; the hardware tests must run on the target GPUs.
