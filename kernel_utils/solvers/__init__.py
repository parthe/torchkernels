import torch

def direct_solve(model, X, y, reg=0.):
    kmat = model.kernel_matrix(X)
    if (model.centers is X) or (reg==0):
        return torch.linalg.lstsq(kmat + reg*X.shape[0], y).solution
    else:
        A = kmat.T @ kmat
        b = kmat.T @ y
        return torch.linalg.lstsq(A + reg*torch.eye(A.shape[-1]), b).solution
