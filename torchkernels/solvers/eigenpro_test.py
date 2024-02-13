import math
import torch

from . import lstsq
from .eigenpro import eigenpro_solver
from ..kernels.radial import LaplacianKernel
from ..linalg.fmm import KmV

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

K = LaplacianKernel(bandwidth=1.)
n, d, c, q = 1000, 3, 2, 30
X = torch.randn(n, d)/math.sqrt(d)
y = torch.randn(n, c)
ahat = eigenpro_solver(K, X, y, q, epochs=100)
astar = lstsq(K, X, X, y, verbose=True)
print((KmV(K, X, X, ahat) - y).var())
