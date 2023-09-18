from torchkernels.linalg.eigh import top_eigensystem, nystrom_extension
from torchkernels.linalg.fmm import KmV
from torchkernels.linalg.rp_cholesky import rp_cholesky_sampler
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
    E, L, lqp1, beta = top_eigensystem(K, X[nids], q)
    E.mul_(((1-lqp1/L)/L/len(nids)).sqrt())
    H = K(X, X[nids]) @ E
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta/lqp1) + 1
    if m is None: m = bs_crit 
    η = cache(lambda m: 1/beta if m < bs_crit else 2/(beta+(m-1)*lqp1))
    
    print(f"bs_crit={bs_crit}, m={m}, η={η(m).item()}")
    timer.toc("EigenPro_2 Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - η(len(bids)) * gm
            a[nids] += η(len(bids)) *  E @ (H[bids].T  @ gm)
    timer.toc("EigenPro_2 Iterations :")
    return a


def eigenpro2_rpc(K, X, y, nids, q, m=None, epochs=1):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    timer.tic()
    n = X.shape[0]
    E, L, lqp1, beta = top_eigensystem(K, X[nids], q)
    E.mul_(((1-lqp1/L)/L/len(nids)).sqrt())
    H = K(X, X[nids]) @ E
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta/lqp1) + 1
    if m is None: m = bs_crit 
    η = cache(lambda m: 1/beta if m < bs_crit else 2/(beta+(m-1)*lqp1))
    
    print(f"bs_crit={bs_crit}, m={m}, η={η(m).item()}")
    timer.toc("EigenPro_2_rpc Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - η(len(bids)) * gm
            a[nids] += η(len(bids)) *  E @ (H[bids].T  @ gm)
    timer.toc("EigenPro_2_rpc Iterations :")
    return a



if __name__ == "__main__":

    from torchkernels.kernels.radial import LaplacianKernel
    from __init__ import lstsq
    import torch
    
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    K = LaplacianKernel(bandwidth=1.)
    n, d, c, s, q = 1000, 3, 2, 100, 5
    epochs = 100

    X = torch.randn(n, d)
    y = torch.randn(n, c)
    
    astar = lstsq(K, X, X, y)
    print((KmV(K,X,X,astar)-y).var())

    ahat1 = eigenpro2(K, X, y, s, q, epochs=epochs)
    print((KmV(K,X,X,ahat1)-y).var())
    
    _,_, nids = rp_cholesky_sampler(K, X, subsample_size=s, alg='rp')
    ahat2 = eigenpro2_rpc(K, X, y, nids, q, epochs=epochs)
    print((KmV(K,X,X,ahat2)-y).var())
    