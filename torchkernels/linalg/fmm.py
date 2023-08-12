# Functional matrix multiplication
#    1. multiplies two matrices in functional form
#        e.g.: 
#            if  A_ij = f(xi, xj) 
#            and B_jk = g(xj, xk)
#            then (A B)_ik = sum_j f(xi, xj) * g(xj, xk)
#        can be calculated without storing the entire A and B matrices
#     2. multiplies a matrix with a vector
#     3. multiplies a gram-matrix with a vector

import torch, math

def fmm(f1, f2, X, Y, Z, out=None, row_chunk_size=None, col_chunk_size=None, mid_chunk_size=None):
    """
        calculate matrix multiplication of f1(X, Y) @ f2(Y, Z) without storing entire matrices
        If argument `out` is provided, the result is added to `out`
    """
    n_r, n_m, n_c = len(X), len(Y), len(Z)
    b_r = n_r if row_chunk_size is None else row_chunk_size
    b_m = n_m if mid_chunk_size is None else mid_chunk_size
    b_c = n_c if col_chunk_size is None else col_chunk_size
    
    return_flag = False
    if out is None:
        return_flag = True
        out = torch.zeros(n_r, n_c)

    for i in range(math.ceil(n_r/b_r)):
        for k in range(math.ceil(n_c/b_c)):
             for j in range(math.ceil(n_m/b_m)):
                out[i*b_r:(i+1)*b_r, k*b_c:(k+1)*b_c] += f1(X[i*b_r:(i+1)*b_r], Y[j*b_m:(j+1)*b_m]) @ f2(Y[j*b_m:(j+1)*b_m], Z[k*b_c:(k+1)*b_c])

    if return_flag: return out

def KmV(K, X, Z, v, out=None, row_chunk_size=None, col_chunk_size=None):
    """
        calculate kernel matrix vector product K(X, Z) @ v without storing kernel matrix
        If argument `out` is provided, the result is added to `out`
    """
    n_r, n_c = len(X), len(Z)
    b_r = n_r if row_chunk_size is None else row_chunk_size
    b_c = n_c if col_chunk_size is None else col_chunk_size
    return_flag = False
    if out is None:
        return_flag = True
        out = torch.zeros(n_r, *v.shape[1:])

    for i in range(math.ceil(n_r/b_r)):
        for j in range(math.ceil(n_c/b_c)):
             out[i*b_r:(i+1)*b_r] += K(X[i*b_r:(i+1)*b_r], Z[j*b_c:(j+1)*b_c]) @ v[j*b_c:(j+1)*b_c]

    if return_flag: return out

def KtKmV(K, X, Z, v, out=None, row_chunk_size=None, col_chunk_size=None):
    """
        calculate kernel matrix vector product K(Z, X) @ K(X, Z) @ v without storing kernel matrix
        If argument `out` is provided, the result is added to `out`

        Note: Currently calls KmV twice. Can be optimized further.
    """
    mid = KmV(K, X, Z, v, row_chunk_size=row_chunk_size, col_chunk_size=col_chunk_size)
    return KmV(K, Z, X, mid, out=out, row_chunk_size=col_chunk_size, col_chunk_size=row_chunk_size)
