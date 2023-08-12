from torchkernels.linalg.fmm import KmV 
from pytictoc import TicToc; timer = TicToc()

def conjugate_gradient(K, X, y, epochs=None):

    timer.tic()
    r = y.type(X.type())
    p = r*1.
    x = torch.zeros_like(y, dtype=X.dtype)
    if epochs is None: epochs = X.shape[0]
    for t in range(epochs):
        Kp = KmV(K,X,X,p)
        r_norm2 = r.pow(2).sum(0)
        alpha = r_norm2/(p * Kp).sum(0)
        x += alpha * p
        r -= alpha * Kp
        beta = r.pow(2).sum(0)/r_norm2
        p = r + beta*p
        print(t, r.var().log10().item())
    timer.toc("Conjugate Gradient Iterations :")
    return x

if __name__ == "__main__":
    import torch
    torch.set_default_dtype(torch.float32)
    from torchkernels.kernels.radial import LaplacianKernel
    K = LaplacianKernel(bandwidth=1.)
    X = torch.randn(100,3)
    y = torch.randn(100,2)
    conjugate_gradient(K, X, y)
    astar = torch.linalg.lstsq(K(X,X), y).solution
    print((KmV(K, X, X, astar) - y).var())