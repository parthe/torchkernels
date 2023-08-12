import torch
from math import ceil
from torchkernels.linalg.fmm import KmV
from __init__ import timer


def mass(K, X, y, m=None, epochs=1):
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
    b = torch.zeros_like(y, dtype=kmat.dtype)
    d = torch.zeros_like(y, dtype=kmat.dtype)
    bs_crit = int(beta/l1) + 1
    if m is None: m = bs_crit 
    η1 = lambda bs: 1/beta if bs < bs_crit else 2/(beta+(bs-1)*l1)
    η2 = lambda bs: 1
    γ = lambda bs: 0.9
    mse = torch.zeros(epochs*ceil(n/m))
    print(f"bs_crit={bs_crit}, m={m}, η1={η1(m)}, η2={η2(m)}, γ={γ(m)}")
    timer.toc("MaSS Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, b) - y[bids].type(b.type())
            d.mul_(γ(len(bids)))
            d[bids] += (η2(len(bids)) - (1 + γ(len(bids)))*η1(len(bids)))*gm
            if t+i>0: # skip for 1st step
                d[bids_] += γ(len(bids)) * η1(len(bids)) * gm_
            gm_, bids_ = gm.clone(), bids.clone()
            b.add_(d)
    timer.toc("MaSS Iterations :")
    return b


if __name__ == "__main__":

    from torchkernels.kernels.radial import LaplacianKernel
    from __init__ import lstsq
    import torch

    torch.set_default_dtype(torch.float64)

    K = LaplacianKernel(bandwidth=1.)
    n, d, c = 100, 3, 2
    X = torch.randn(n, d)
    y = torch.randn(n, c)
    ahat = mass(K, X, y, epochs=100)
    astar = lstsq(K, X, X, y)
    print((KmV(K, X, X, ahat) - y).var())
    print((KmV(K, X, X, astar) - y).var())


