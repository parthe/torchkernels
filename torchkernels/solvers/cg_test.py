import torch, math
from . import lstsq
from .cg import conjugate_gradient
from ..linalg.fmm import KmV
from ..kernels.radial import LaplacianKernel

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

K = LaplacianKernel(bandwidth=1.)
n, d, c = 1000, 3, 1
X = torch.randn(n, d)
y = torch.randn(n, c)
ahat = conjugate_gradient(K, X, y, epochs=n)
astar = lstsq(K, X, X, y)
print((KmV(K, X, X, ahat) - y).var())
print((KmV(K, X, X, astar) - y).var())