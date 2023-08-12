from torchkernels.linalg.fmm import KmV 
from __init__ import timer


def conjugate_gradient(K, X, y, epochs=None):

    timer.tic()
    a = torch.zeros_like(y, dtype=X.dtype)
    r = y.type(X.type()).clone()
    p = r.clone()
    if epochs is None: epochs = X.shape[0]
    for t in range(epochs):
        Kp = KmV(K, X, X, p)
        r_norm2 = r.pow(2).sum(0)
        alpha = r_norm2/(p * Kp).sum(0)
        a += alpha * p
        r -= alpha * Kp
        beta = r.pow(2).sum(0)/r_norm2
        p = r + beta*p
    timer.toc("Conjugate Gradient Iterations :")
    return a

if __name__ == "__main__":

    import torch, math
    from __init__ import lstsq
    from torchkernels.kernels.radial import LaplacianKernel

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    K = LaplacianKernel(bandwidth=1.)
    n, d, c = 1000, 3, 1
    X = torch.randn(n, d)
    y = torch.randn(n, c)
    ahat = conjugate_gradient(K, X, y, epochs=n)
    astar = lstsq(K, X, X, y)
    print((KmV(K, X, X, ahat) - y).var())
    print((KmV(K, X, X, astar) - y).var())