import torch

eps = 1e-12

def norm(X, squared=False, M=None, keepdim=False, in_place=True):
    '''Calculate the norm.
    If `M` is not None, then Mahalanobis norm is calculated.

    Args:
        X: of shape (n, d).
        M: of shape (d, d). positive semi-definite matrix for Mahalanobis norm.
        squared: boolean.
        in_place: boolean.

    Returns:
        pointwise norm (n,).
    '''
    if M is None:
        X_norm = X.pow(2).sum(dim=-1, keepdim=keepdim)
    else:
        X_norm = ((X @ M) * X).sum(dim=-1, keepdim=keepdim)
    if squared:
        return X_norm
    else:
        if in_place:
            X_norm.clamp_(min=0).sqrt_()
            return X_norm
        else:
            return X_norm.clamp_(min=0).sqrt_()


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

def euclidean(samples, centers, squared=False, M=None, in_place=True):
    '''Calculate the pairwise euclidean distance.
     If `M` is not None, then Mahalanobis distance is calculated.

    Args:
        samples: of shape (n, d).
        centers: of shape (p, d).
        squared: boolean.
        M: of shape (d, d). positive semi-definite matrix for Mahalanobis norm.
        in_place: boolean.

    Returns:
        pointwise distances (n, p).
    '''
    samples_norm2 = norm(samples, squared=True, M=M, keepdim=True, in_place=in_place)
    centers_norm2 = samples_norm2.T if (samples is centers) else norm(centers, squared=True, M=M, keepdim=False, in_place=in_place)
    
    distances = inner_product(samples, centers, M=M)

    if in_place:
        distances.mul_(-2)
        distances.add_(samples_norm2)
        distances.add_(centers_norm2)
        distances.clamp_(min=0)
        if not squared:
            distances.sqrt_()
        return distances
    else:
        if not squared:
            return torch.clamp(samples_norm2+centers_norm2-2*distances,min=0).sqrt()
        else:
            return torch.clamp(samples_norm2+centers_norm2-2*distances,min=0)
