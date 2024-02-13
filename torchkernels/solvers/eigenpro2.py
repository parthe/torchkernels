from ..linalg.eigh import top_eigensystem, nystrom_extension
from ..linalg.fmm import KmV
from ..linalg.rp_cholesky import rp_cholesky_sampler
from torchmetrics.functional import mean_squared_error as mse
from ..utils import timer
from math import ceil
import torch
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
    E.mul_(((1 - lqp1 / L) / L).sqrt())
    H = K(X, X[nids]) @ E
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta * s / lqp1) + 1
    if m is None: m = bs_crit
    lr = cache(lambda m: 1 / beta if m < bs_crit else 2 / (beta + (m - 1) * lqp1/s))

    print(f"bs_crit={bs_crit}, m={m}, lr={lr(m).item()}")
    timer.toc("EigenPro_2 Setup :", restart=True)
    err = torch.ones(epochs) * torch.nan
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - lr(len(bids)) * gm
            a[nids] += lr(len(bids)) * E @ (H[bids].T @ gm)
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
    s = len(nids)
    E, L, lqp1, beta = top_eigensystem(K, X[nids], q, method="scipy.linalg.eigh")
    E.mul_(((1 - lqp1 / L) / L).sqrt())
    H = K(X, X[nids]) @ E
    a = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta * s / lqp1) + 1
    if m is None: m = bs_crit
    lr = cache(lambda m: 1 / beta if m < bs_crit else 2 / (beta + (m - 1) * lqp1/s))

    print(f"bs_crit={bs_crit}, m={m}, lr={lr(m).item()}")
    timer.toc("EigenPro_2_rpc Setup :", restart=True)
    err = torch.ones(epochs) * torch.nan
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = KmV(K, X[bids], X, a) - y[bids].type(a.type())
            a[bids] = a[bids] - lr(len(bids)) * gm
            a[nids] += lr(len(bids)) * E @ (H[bids].T @ gm)
        err[t] = mse(KmV(K, X, X, a), y)
    timer.toc("EigenPro_2_rpc Iterations :")
    return a, err


