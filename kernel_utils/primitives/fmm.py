import torch, math

def KmV(K, X, Z, v, out=None, row_chunk_size=None, col_chunk_size=None):
    """
        calculate kernel matrix vector product K(X, Z) @ v without storing kernel matrix
        If argument `out` is provided, the result is added to `out`
    """
    n_r, n_c = len(X), len(Z)
    b_r = n_r if row_chunk_size is None else row_chunk_size
    b_c = n_c if col_chunk_size is None else col_chunk_size

    if out is None:
        out = torch.zeros(n_r, *v.shape[1:])

    for i in range(math.ceil(n_r/b_r)):
        for j in range(math.ceil(n_c/b_c)):
             out[i*b_r:(i+1)*b_r] += K(X[i*b_r:(i+1)*b_r], Z[j*b_c:(j+1)*b_c]) @ v[j*b_c:(j+1)*b_c]

    if out is None:
        return out

def KtKmV(K, X, Z, v, out=None, row_chunk_size=None, col_chunk_size=None):
    """
        calculate kernel matrix vector product K(Z, X) @ K(X, Z) @ v without storing kernel matrix
        If argument `out` is provided, the result is added to `out`

        Note: Currently calls KmV twice. Can be optimized further.
    """
    mid = KmV(K, X, Z, v, row_chunk_size=row_chunk_size, col_chunk_size=col_chunk_size)
    return KmV(K, Z, X, mid, out=out, row_chunk_size=col_chunk_size, col_chunk_size=row_chunk_size)
