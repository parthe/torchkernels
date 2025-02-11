import torch
from torchkernels.linalg.eigh import top_eigensystem
from torchkernels.linalg.fmm import KmV
from torchkernels.kernels.radial import LaplacianKernel
from torchkernels.solvers import lstsq, timer
from torchkernels.solvers.eigenpro import eigenpro
from math import ceil
from functools import cache

steps = 3


def multistep_richardson(K, X, y, m=None, epochs=1, steps=3):
    timer.tic()
    n = X.shape[0]
    E, L, lqp1, beta = top_eigensystem(K, X, 1)
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta/lqp1) + 1
    if m is None: m = bs_crit 
    η  = cache(lambda m: 1/beta if m < bs_crit else 2/(beta+(m-1)*lqp1))

    print(f"bs_crit={bs_crit}, m={m}, η={η(m).item()}")
    timer.toc("Multi EigenPro Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            sgm = torch.zeros_like(y[bids], dtype=a.dtype)
            for step in range(steps):
                if step==0:
                    gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
                else:
                    gm -= η(len(bids)) * KmV(K, X[bids], X[bids], gm)
                sgm += gm
            a.index_add_(0, bids, η(len(bids)) * sgm, alpha=-1)
        # print(t, (Kmat @ a - y).var())
    timer.toc(f"Multi Richardson Iterations ({steps} steps):")
    return a



def multistep_eigenpro(K, X, y, q, m=None, epochs=1, steps=3):

    timer.tic()
    n = X.shape[0]
    E, L, lqp1, beta = top_eigensystem(K, X, q)
    F = E * (1-lqp1/L).sqrt()
    H = KmV(K, X, X, F)
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta/lqp1) + 1
    if m is None: m = bs_crit 
    η  = cache(lambda m: 1/beta if m < bs_crit else 2/(beta+(m-1)*lqp1))

    print(f"bs_crit={bs_crit}, m={m}, η={η(m).item()}")
    timer.toc("Multi EigenPro Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            sgm = torch.zeros_like(y[bids], dtype=a.dtype)
            for step in range(steps):
                if step==0:
                    gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
                else:
                    gm -= η(len(bids)) * (KmV(K, X[bids], X[bids], gm)) - η(len(bids)) * (H[bids]  @ (F[bids].T @ gm))
                sgm += gm
            a.index_add_(0, bids, η(len(bids)) * sgm, alpha=-1)
            a += η(len(bids)) *  F @ (F[bids].T  @ sgm)    
        # print(t, (Kmat @ a - y).var())
    timer.toc(f"Multi EigenPro Iterations ({steps} steps):")
    return a


if __name__ == "__main__":

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    K = LaplacianKernel(length_scale=1.)
    n, d, c, q, m = 100, 3, 2, 1, None
    X = torch.randn(n, d)
    y = torch.randn(n, c)
    ahat1 = multistep_eigenpro(K, X, y, q, epochs=n)
    print((KmV(K, X, X, ahat1) - y).var())
    ahat2 = eigenpro(K, X, y, q, epochs=n)
    print((KmV(K, X, X, ahat2) - y).var())
    astar = lstsq(K, X, X, y)
    print((KmV(K, X, X, astar) - y).var())
