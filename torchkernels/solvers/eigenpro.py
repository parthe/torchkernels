import torch, os
from ..linalg.eigh import top_eigensystem, nystrom_extension
from ..linalg.fmm import KmV
from . import timer
from functools import cache


def eigenpro_solver(K, X, y, q, m=None, epochs=1):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    timer.tic()
    n = X.shape[0]
    E, L, lqp1, beta = top_eigensystem(K, X, q)
    E.mul_((1 - lqp1 / L).sqrt())
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta * n / lqp1) + 1
    if m is None: m = bs_crit
    lr = cache(lambda m: 1 / beta if m < bs_crit else 2 / (beta + (m - 1) * lqp1 / n))

    print(f"bs_crit={bs_crit}, m={m}, η={lr(m).item()}")
    timer.toc("EigenPro Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - lr(len(bids)) * gm
            a += lr(len(bids)) * E @ (E[bids].T @ gm)
    timer.toc("EigenPro Iterations :")
    return a
