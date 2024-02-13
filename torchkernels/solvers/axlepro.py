from torchkernels.linalg.eigh import top_eigensystem, nystrom_extension
from torchkernels.linalg.fmm import KmV
from torchmetrics.functional import mean_squared_error as mse
from ..utils import timer
from math import ceil, sqrt
import torch
import scipy
from functools import cache, partial


def hyperparameter_selection(m, n, beta, lqp1, lam_min):
    # assumes lqp1 and lam_min are normalized
    mu = lam_min / n
    ktil_m = n / m + (m - 1) / m
    Lm = (beta + (m - 1) * lqp1) / m
    k_m = Lm / mu
    eta_1 = 1 / Lm
    t_ = sqrt(k_m * ktil_m)
    eta_2 = ((eta_1 * t_) / (t_ + 1)) * (1 - 1 / ktil_m)
    gamma = (t_ - 1) / (t_ + 1)
    return eta_1/m, eta_2/m, gamma



def lm_axlepro(K, X, y, s, q, m=None, epochs=1):
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
    a = torch.zeros_like(y, dtype=E.dtype)
    b = torch.zeros_like(y, dtype=E.dtype)
    bs_crit = int(beta * s / lqp1) + 1
    m = bs_crit if m is None else m
    mu = scipy.linalg.eigh(K(X[nids], X[nids]),
                           eigvals_only=True, subset_by_index=[0, 0])[0]
    print(mu)
    lrs = cache(partial(hyperparameter_selection,
                        n=n, beta=beta, lqp1=lqp1 / s, lam_min=mu / s))
    lr1, lr2, damp = lrs(m)
    print(f"bs_crit={bs_crit}, m={m}, lr1={lr1.item()}, "
          f"lr2={lr2.item()}, damp={damp}")
    timer.toc("LM-AxlePro Setup :", restart=True)
    err = torch.ones(epochs) * torch.nan
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            lr1, lr2, damp = lrs(len(bids))
            Km = K(X[bids], X)
            v = Km @ b - y[bids].type(a.type())
            w = E @ (E.T @ (Km.T[nids] @ v))
            a_ = a.clone()
            a = b
            a[bids] -= lr1 * v
            a[nids] += lr1 * w
            b = (1 + damp) * a - damp * a_
            b[bids] += lr2 * v
            b[nids] -= lr2 * w
        err[t] = mse(KmV(K, X, X, a), y)
    timer.toc("LM-AxlePro Iterations :")
    return a, err
