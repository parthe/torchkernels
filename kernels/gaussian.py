import primitives

def gaussian(samples, centers, bandwidth=1., M=None):
    '''Gaussian kernel.
    If `M` is not None, then Mahalanobis metric is applied.

    Args:
        samples: of shape (n, d).
        centers: of shape (p, d).
        bandwidth: kernel bandwidth.
        M: of shape (d, d). positive semi-definite matrix for Mahalanobis norm.

    Returns:
        kernel matrix of shape (n, p).
    '''
    assert bandwidth > 0
    kernel_mat = primitives.euclidean(samples, centers, squared=True, M=M)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat
