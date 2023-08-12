from torchkernels.linalg.eigh import top_eigensystem, nystrom_extension
from torchkernels.linalg.fmm import KmV
from __init__ import timer
from math import ceil
from functools import cache



def eigenpro(K, X, y, q, m=None, epochs=1):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    timer.tic()
    n = X.shape[0]
    E, L, lqp1, beta = top_eigensystem(K, X, q)
    E.mul_((1-lqp1/L).sqrt())
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta/lqp1) + 1
    if m is None: m = bs_crit 
    @cache
    def η(m): return 1/beta if m < bs_crit else 2/(beta+(m-1)*lqp1)
    print(f"bs_crit={bs_crit}, m={m}, η={η(m).item()}")
    timer.toc("EigenPro Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - η(len(bids)) * gm
            a += η(len(bids)) *  E @ (E[bids].T  @ gm)
    timer.toc("EigenPro Iterations :")
    return a

if __name__ == "__main__":

    from torchkernels.kernels.radial import LaplacianKernel
    from __init__ import lstsq
    import torch

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    
    K = LaplacianKernel(bandwidth=1.)
    n, d, c, q = 100, 3, 2, 5
    X = torch.randn(n, d)
    y = torch.randn(n, c)
    ahat = eigenpro(K, X, y, q, epochs=n)
    astar = lstsq(K, X, X, y)
    print((KmV(K,X,X,ahat)-y).var())
    print((KmV(K,X,X,astar)-y).var())
