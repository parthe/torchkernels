from torchkernels.linalg.eigh import top_eigensystem, nystrom_extension
from torchkernels.linalg.fmm import KmV
from torchkernels.linalg.rp_cholesky import rp_cholesky_sampler
from torchmetrics.functional import mean_squared_error as mse
from __init__ import timer
from math import ceil
from functools import cache


def eigenpro2(K, X, y, s, q, m=None, epochs=1):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    timer.tic()
    n = X.shape[0]
    nids = torch.randperm(n)[:s]
    E, L, lqp1, beta = top_eigensystem(K, X[nids], q, method="scipy.linalg.eigh")
    E.mul_(((1-lqp1/L)/L/len(nids)).sqrt())
    H = K(X, X[nids]) @ E
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta/lqp1) + 1
    if m is None: m = bs_crit 
    η = cache(lambda m: 1/beta if m < bs_crit else 2/(beta+(m-1)*lqp1))
    
    print(f"bs_crit={bs_crit}, m={m}, η={η(m).item()}")
    timer.toc("EigenPro_2 Setup :", restart=True)
    err = torch.ones(epochs)*torch.nan
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - η(len(bids)) * gm
            a[nids] += η(len(bids)) *  E @ (H[bids].T  @ gm)
        err[t] = mse(KmV(K, X, X, a), y)
    timer.toc("EigenPro_2 Iterations :")
    return a, err


def eigenpro2_rpc(K, X, y, nids, q, m=None, epochs=1):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    timer.tic()
    n = X.shape[0]
    E, L, lqp1, beta = top_eigensystem(K, X[nids], q, method="scipy.linalg.eigh")
    E.mul_(((1-lqp1/L)/L/len(nids)).sqrt())
    H = K(X, X[nids]) @ E
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta/lqp1) + 1
    if m is None: m = bs_crit 
    η = cache(lambda m: 1/beta if m < bs_crit else 2/(beta+(m-1)*lqp1))
    
    print(f"bs_crit={bs_crit}, m={m}, η={η(m).item()}")
    timer.toc("EigenPro_2_rpc Setup :", restart=True)
    err = torch.ones(epochs)*torch.nan
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - η(len(bids)) * gm
            a[nids] += η(len(bids)) *  E @ (H[bids].T  @ gm)
        err[t] = mse(KmV(K, X, X, a), y)
    timer.toc("EigenPro_2_rpc Iterations :")
    return a, err



if __name__ == "__main__":

    from torchkernels.kernels.radial import LaplacianKernel
    import matplotlib.pyplot as plt
    from __init__ import lstsq
    import torch
    
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    K = LaplacianKernel(bandwidth=1.)
    n, d, c, s, q = 1000, 3, 2, 20, 5
    epochs = 500

    X = torch.randn(n, d)
    y = torch.randn(n, c)
    
    astar = lstsq(K, X, X, y)
    err0 = mse(KmV(K,X,X,astar),y)
    print(err0)

    ahat1, err1 = eigenpro2(K, X, y, s, q, epochs=epochs)
    print(err1[-1])
    
    _,_, nids = rp_cholesky_sampler(K, X, subsample_size=s, alg='rp')
    ahat2, err2 = eigenpro2_rpc(K, X, y, nids, q, epochs=epochs)
    print(err2[-1])
    
    plt.plot(err1, 'b', label='random')
    plt.plot(err2, 'g', label='cholesky')
    plt.hlines(err0, 0, epochs, linestyles='dashed', colors='k')
    plt.yscale('log')
    plt.title(f'Nyström subset size = {s}')
    plt.legend()
    plt.savefig(f'Nyström subset size = {s}.png')
    plt.show()
