import torch
from pytictoc import TicToc; timer = TicToc()


def lstsq(K, X, Z=None, y=None, reg=0.):
    kmat = K(X, Z)
    timer.tic()
    if y is None: 
        raise ValueError("argument `y` must not be None")
    if (Z is None) or (reg==0):
        ahat = torch.linalg.lstsq(kmat + reg*X.shape[0], y).solution
    else:
        A = kmat.T @ kmat
        b = kmat.T @ y
        ahat = torch.linalg.solve(A + reg*torch.eye(A.shape[-1]), b)
    timer.toc("Direct solve :")
    return ahat
