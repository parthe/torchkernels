import torch
from math import ceil
from torchkernels.linalg.fmm import KmV
from __init__ import timer
from functools import cache


def richardson(K, X, y, m=None, epochs=1):
    """
        Storage: (n x m)
        FLOPS at setup: (n) 
        FLOPS per batch: (n x m)
    """
    timer.tic()
    n = X.shape[0]
    kmat = K(X)
    lam_1, _ = torch.lobpcg(kmat/n, 1); print(lam_1.dtype, lam_1.item())
    beta = kmat.diag().max()
    a = torch.zeros_like(y, dtype=kmat.dtype)
    bs_crit = int(beta/lam_1) + 1
    if m is None: m = bs_crit 
    η = cache(lambda m: 1/beta if m < bs_crit else 2/(beta+(m-1)*lam_1))
    
    print(f"bs_crit={bs_crit}, m={m}, η={η(m).item()}")
    timer.toc("Richardson Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - η(len(bids)) * gm
    timer.toc("Richardson Iterations :")
    return a

if __name__ == "__main__":

    from torchkernels.kernels.radial import LaplacianKernel
    from __init__ import lstsq
    import torch

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    K = LaplacianKernel(bandwidth=1.)
    n, d, c = 100, 3, 2
    X = torch.randn(n, d)
    y = torch.randn(n, c)
    ahat = richardson(K, X, y, epochs=n)
    astar = lstsq(K, X, X, y)
    print((KmV(K, X, X, ahat) - y).var())
    print((KmV(K, X, X, astar) - y).var())
