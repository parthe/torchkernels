import torch
from math import ceil
from pytictoc import TicToc; timer = TicToc()

def batched(K, X, y, q, m=None, epochs=1, return_error=True, seed=None):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    if seed is not None: torch.manual_seed(seed)
    timer.tic()
    n = X.shape[0]
    K(X, X, save=True)
    l1, _ torch.lobpcg(K.matrix, 1,)
    beta = K.matrix.diag().max()
    a = torch.zeros_like(y, dtype=K.matrix.dtype)
    bs_crit = int(beta/l1) + 1
    if m is None: m = bs_crit 
    lr = lambda bs: 1/beta if bs < bs_crit else 2/(beta+(bs-1)*l1)
    mse = torch.zeros(epochs*ceil(n/m))
    print(f"bs_crit={bs_crit}, m={m}, lr={lr(m)}")
    timer.toc("EigenPro1 Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = K.matrix[bids] @ a - y[bids].type(a.type())
            a[bids] = a[bids] - lr(len(bids)) * gm
            if return_error: mse[t*len(batches)+i] = (K.matrix @ a - y).var()
    timer.toc("EigenPro1 Iterations :")
    return (a, None) if not return_error else (a, mse)
