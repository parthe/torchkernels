import torch
from math import ceil
from torchkernels.linalg.fmm import KmV
from __init__ import timer


def richardson(K, X, y, m=None, epochs=1):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    timer.tic()
    n = X.shape[0]
    kmat = K(X)
    l1, _ = torch.lobpcg(kmat/n, 1)
    beta = kmat.diag().max()
    a = torch.zeros_like(y, dtype=kmat.dtype)
    bs_crit = int(beta/l1) + 1
    if m is None: m = bs_crit 
    lr = lambda bs: 1/beta if bs < bs_crit else 2/(beta+(bs-1)*l1)
    mse = torch.zeros(epochs*ceil(n/m))
    print(f"bs_crit={bs_crit}, m={m}, lr={lr(m)}")
    timer.toc("EigenPro1 Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - lr(len(bids)) * gm
    timer.toc("EigenPro1 Iterations :")
    return a

if __name__ == "__main__":

    from torchkernels.kernels.radial import LaplacianKernel
    from __init__ import lstsq
    import torch

    torch.set_default_dtype(torch.float64)

    K = LaplacianKernel(bandwidth=1.)
    n, d, c = 100, 3, 2
    X = torch.randn(n, d)
    y = torch.randn(n, c)
    ahat = richardson(K, X, y, epochs=100)
    astar = lstsq(K, X, X, y)
    print((KmV(K, X, X, ahat) - y).var())
    print((KmV(K, X, X, astar) - y).var())
