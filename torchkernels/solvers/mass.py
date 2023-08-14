import torch
from math import ceil
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
    a, b = torch.zeros_like(y, dtype=kmat.dtype), torch.zeros_like(y, dtype=kmat.dtype)
    bs_crit = int(beta/lam_1) + 1
    if m is None: m = bs_crit 

    L = cache(lambda m: beta/m + (m-1)*lam_1/m)
    kappa = cache(lambda m: L(m)/lam_n)
    kappa_til = cache(lambda m: n/m + (m - 1)/m)
    sqrt_κm_κm_til = cache(lambda m: torch.sqrt(kappa(m) * kappa_til(m)))
    η1 = cache(lambda m: 1/L(m))
    η2 = cache(lambda m: ((η1(m) * sqrt_κm_κm_til(m)) / (sqrt_κm_κm_til(m) + 1)) * (1 - 1/kappa_til(m)))
    γ = cache(lambda m: (sqrt_κm_κm_til(m) - 1) / (sqrt_κm_κm_til(m) + 1))

    print(f"bs_crit={bs_crit}, m={m}, η1={η1(m).item()}, "
        f"η2={η2(m).item()}, γ={γ(m)}")
    timer.toc("MaSS Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            # gm = KmV(K, X[bids], X, b) - y[bids].type(b.type())
            # d.mul_(γ(len(bids)))
            # d[bids] += (η2(len(bids)) - (1 + γ(len(bids)))*η1(len(bids)))*gm/len(bids)
            # if t+i>0: # skip for 1st step
            #    d[bids_] += γ(len(bids)) * η1(len(bids)) * gm_/len(bids_)
            # gm_, bids_ = gm.clone(), bids.clone()
            # b.add_(d)
            gm = KmV(K, X[bids],X, b) - y[bids].type(b.type())
            a_ = a.clone()
            a = b
            a[bids] -= η1(m)*gm/len(bids)
            b = a + γ(m)*(a-a_)
            b[bids] += η2(m)*gm/len(bids)
        print(t, (kmat @ b - y).var())
    timer.toc("MaSS Iterations :")
    return b


if __name__ == "__main__":

    from torchkernels.kernels.radial import LaplacianKernel
    from __init__ import lstsq
    import torch

    torch.set_default_dtype(torch.float64)

    K = LaplacianKernel(bandwidth=1.)
    n, d, c = 1000, 3, 2
    X = torch.randn(n, d)
    y = torch.randn(n, c)
    ahat = mass(K, X, y, epochs=100)
    astar = lstsq(K, X, X, y)
    print((KmV(K, X, X, ahat) - y).var())
    print((KmV(K, X, X, astar) - y).var())


