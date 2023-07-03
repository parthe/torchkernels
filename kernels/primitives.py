import torch

eps = 1e-12

def norm(X, squared=False, M=None):
    '''Calculate the norm.
    If `M` is not None, then Mahalanobis norm is calculated.

    Args:
        X: of shape (n, d).
        M: of shape (d, d). positive semi-definite matrix for Mahalanobis norm.
        squared: boolean.

    Returns:
        pointwise norm (n,).
    '''
    if M is None:
        X2 = torch.sum(X**2, dim=1) if squared:    
        return X2 if squared else X2.sqrt()
    else:
        XMX = ((X @ M) * X).sum(dim=-1)
        return XMX if squared else XMX.sqrt()

def inner_product(samples, centers, M=None):
    '''Calculate the pairwise inner-product.
     If `M` is not None, then Mahalanobis inner-product is calculated

    Args:
        samples: of shape (n, d).
        centers: of shape (p, d).
        M: of shape (d, d). positive semi-definite matrix for Mahalnobis inner-product.

    Returns:
        pointwise distances (n, p).
    '''
    return samples @ centers.T if M is None else samples @ M @ centers.T

def euclidean(samples, centers, squared=False, M=None):
    '''Calculate the pairwise euclidean distance.
     If `M` is not None, then Mahalanobis distance is calculated.

    Args:
        samples: of shape (n, d).
        centers: of shape (p, d).
        squared: boolean.
        M: of shape (d, d). positive semi-definite matrix for Mahalanobis norm.

    Returns:
        pointwise distances (n, p).
    '''
    samples_norm2 = norm(samples, squared=True, M=M)
    centers_norm2 = samples_norm2 if samples is centers else norm(centers, squared=True, M=M)
    
    distances = inner_product(samples, centers, M=M)
    distances.mul_(-2)
    distances.add_(samples_norm2.view(-1, 1))
    distances.add_(centers_norm2)

    return distances if squared else distances.clamp_(min=0).sqrt_()
