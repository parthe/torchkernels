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
        X2 = torch.sum(X**2, dim=-1).clamp(min=0)
        return X2 if squared else X2.sqrt()
    else:
        XMX = ((X @ M) * X).sum(dim=-1).clamp(min=0)
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
    
    distances2 = ((samples_norm2 if len(centers_norm2.shape)==0 else samples_norm2.unsqueeze(-1)) 
                + centers_norm2 
                - 2 * inner_product(samples, centers, M=M))
    distances2.clamp_(min=0)
    return distances2 if squared else distances2.sqrt()
