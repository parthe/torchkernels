from ..kernels.radial import LaplacianKernel
import matplotlib.pyplot as plt
from ..linalg.fmm import KmV
from .axlepro import lm_axlepro
from .eigenpro2 import eigenpro2
from . import lstsq
import torch
import math
from torchmetrics.functional import mean_squared_error as mse

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)

K = LaplacianKernel(length_scale=1.)
n, d, c, s, q = 1000, 3, 2, 100, 10
epochs = 1000

X = torch.randn(n, d)
y = torch.randn(n, c)

astar = lstsq(K, X, X, y, verbose=True)
err0 = mse(KmV(K, X, X, astar), y)
print(err0)

ahat1, err1 = eigenpro2(K, X, y, s, q, epochs=epochs)
plt.plot(torch.arange(epochs)+1, err1, 'b', label='eigenpro')
print(err1[-1])

ahat2, err2 = lm_axlepro(K, X, y, s, q, epochs=epochs)
plt.plot(torch.arange(epochs)+1, err2, 'g', label='axlepro')
print(err2[-1])

plt.hlines(err0, 0, epochs, linestyles='dashed', colors='k')
plt.yscale('log')
plt.title(f'Nyström subset size = {s}')
plt.legend()
plt.show()