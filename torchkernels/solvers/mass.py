import torch
from math import ceil, sqrt
from torchkernels.linalg.fmm import KmV
from __init__ import timer
from functools import cache


def mass(K, X, y, m=None, epochs=1):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    timer.tic()
    n = X.shape[0]
    kmat = K(X)
    lam_1, _ = torch.lobpcg(kmat/n, 1, largest=True)
    lam_n, _ = torch.lobpcg(kmat/n, 1, largest=False)
    beta = kmat.diag().max()
    b, d = torch.zeros_like(y, dtype=kmat.dtype), torch.zeros_like(y, dtype=kmat.dtype)
    bs_crit = int(beta/lam_1) + 1
    if m is None: m = bs_crit 

    @cache
    def η1(m): return 1/beta if m < bs_crit else 2/(beta+(m-1)*lam_1)

    @cache
    def sqrt_κm_κm_til(m): return sqrt((beta + (m-1)*lam_1)*(m+n-1)/lam_n)/m

    @cache
    def η2(m): return η1(m) * (n-1)/(n+m-1) * 1/(1+1/sqrt_κm_κm_til(m))

    @cache
    def γ(m): return (sqrt_κm_κm_til(m)-1)/(sqrt_κm_κm_til(m)+1)/m

    print(f"bs_crit={bs_crit}, m={m}, η1={η1(m).item()}, "
        f"η2={η2(m).item()}, γ={γ(m)}")
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


