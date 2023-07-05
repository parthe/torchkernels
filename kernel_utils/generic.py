from .primitives import norm, inner_product
import torch, math

def normalized_inner_product(func, x, z, M=None):
    """
        K(x,z) = norm(x) * norm(z) * func(<x_, z_>) where x_ = x/norm(x) and z_ = z/norm(z)
    """
    x_norm = norm(x, M=M)
    z_norm = norm(z, M=M)
    xz = inner_product(x/x_norm.view(-1,1), z/z_norm.view(-1,1), M=M)
    return x_norm.view(-1,1) * func(xz) * z_norm

def fmmv(K, X, Z, v, out=None, row_chunk_size=None, col_chunk_size=None):
    """
        calculate kernel matrix vector product K(X, Z) @ v without storing kernel matrix
    """
    n, p = len(X), len(Z)
    b_n = n if row_chunk_size is None else row_chunk_size
    b_p = p if col_chunk_size is None else col_chunk_size

    if out is None:
        out = torch.zeros(len(X), *v.shape[1:])

    for i in range(math.ceil(n/b_n)):
        for j in range(math.ceil(n/b_n)):
            out[i*b_n:(i+1)*b_n] = K(X[i*b_n:(i+1)*b_n], Z[j*b_p:(j+1)*b_p]) @ v[j*b_p:(j+1)*b_p]

    if out is None:
        return out
        
