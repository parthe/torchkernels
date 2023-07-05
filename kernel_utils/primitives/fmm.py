import torch, math

def KmV(K, X, Z, v, out=None, row_chunk_size=None, col_chunk_size=None):
    """
        calculate kernel matrix vector product K(X, Z) @ v without storing kernel matrix
        If argument `out` is provided, the result is added to `out`
    """
    n, p = len(X), len(Z)
    b_n = n if row_chunk_size is None else row_chunk_size
    b_p = p if col_chunk_size is None else col_chunk_size

    if out is None:
        out = torch.zeros(len(X), *v.shape[1:])

    for i in range(math.ceil(n/b_n)):
        for j in range(math.ceil(n/b_n)):
             out[i*b_n:(i+1)*b_n] += K(X[i*b_n:(i+1)*b_n], Z[j*b_p:(j+1)*b_p]) @ v[j*b_p:(j+1)*b_p]

    if out is None:
        return out
