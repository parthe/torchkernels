from ..kernels.radial import LaplacianKernel
from .eigenpro2 import eigenpro2, eigenpro2_rpc
import matplotlib.pyplot as plt
from torchmetrics.functional import mean_squared_error as mse
from ..linalg.fmm import KmV
from ..linalg.rp_cholesky import rp_cholesky_sampler
from . import lstsq
import torch
import math

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

K = LaplacianKernel(length_scale=1.)
n, d, c, s, q = 1000, 3, 2, 100, 20
epochs = 1000

X = torch.randn(n, d)
y = torch.randn(n, c)

astar = lstsq(K, X, X, y, verbose=True)
err0 = mse(KmV(K, X, X, astar), y)
print(err0)

ahat1, err1 = eigenpro2(K, X, y, s, q, epochs=epochs)
plt.plot(err1, 'b', label='uniform')
print(err1[-1])

_, _, nids = rp_cholesky_sampler(K, X, subsample_size=s, alg='rp')
ahat2, err2 = eigenpro2_rpc(K, X, y, nids, q, epochs=epochs)
plt.plot(err2, 'g', label='rpc')
print(err2[-1])

plt.hlines(err0, 0, epochs, linestyles='dashed', colors='k')
plt.yscale('log')
plt.title(f'Nyström subset size = {s}')
plt.legend()
plt.savefig(f'Nyström subset size = {s}.png')
plt.show()
