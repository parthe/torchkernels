from ..linalg.eigh import top_eigensystem, nystrom_extension

def eigenpro(K, X, y, q, m=None, epochs=1, return_error=True, seed=None):
    """
        Storage: (n x q) + s2
        FLOPS at setup: (s x q2) + 
        FLOPS per batch: (n x m) + {(m x q) + (n x q)}
    """
    if seed is not None: torch.manual_seed(seed)
    timer.tic()
    n = X.shape[0]
    nids = torch.randperm(n)[:q*10]
    val_ids = torch.randperm(n)[:q*10]
    if return_error: K_val = K(X[val_ids], X)
    E, L, lqp1, beta = top_eigensystem(K, X, q)
    F = E * (1-lqp1/L).sqrt()
    del E
    a = torch.zeros_like(y, dtype=F.dtype)
    bs_crit = int(beta/lqp1) + 1
    if m is None: m = bs_crit 
    lr = lambda bs: 1/beta if bs < bs_crit else 2/(beta+(bs-1)*lqp1)
    mse = torch.zeros(epochs*ceil(n/m))
    print(f"bs_crit={bs_crit}, m={m}, lr={lr(m)}")
    timer.toc("EigenPro Setup :", restart=True)
    for t in range(epochs):
        batches = torch.randperm(n).split(m)
        for i, bids in enumerate(batches):
            gm = K(X[bids], X) @ a - y[bids].type(a.type())
            a[bids] = a[bids] - lr(len(bids)) * gm
            a += lr(len(bids)) *  F @ (F[bids].T  @ gm)
            if return_error: mse[t*len(batches)+i] = (K_val @ a - y).var()
    timer.toc("EigenPro Iterations :")
    return (a, None) if not return_error else (a, mse)
