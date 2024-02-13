from torchkernels.linalg.fmm import KmV 
from ..utils import timer
import torch

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
