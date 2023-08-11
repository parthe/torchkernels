import torch

def direct_solve(model, X, y, reg=0.):
    kmat = model.kernel(X, model.centers)
    if (model.centers is X) or (reg==0):
        return torch.linalg.lstsq(kmat + reg*X.shape[0], y).solution
    else:
        A = kmat.T @ kmat
        b = kmat.T @ y
        return torch.linalg.solve(A + reg*torch.eye(A.shape[-1]), b)
