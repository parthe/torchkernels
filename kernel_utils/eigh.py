import torch

def top_eigensystem(K, X, q):
    """
      Top-q eigen system of K(X, X)/n
      where n = len(X)

      Args: 
        K: kernel that takes 2 arguments.
        X: of shape (n, d).
        q: number of eigen-modes

      Returns:
        E: top-q eigenvectors
        L: top-q eigenvalues of
        lqp1: q+1 st eigenvalue
        beta: max{i} of K(xi, xi)
    """
  
    n = X.shape[0]
    scaled_kmat = K(X, X)/n
    L, E = torch.lobpcg(scaled_kmat, q)
    beta = n * scaled_kmat.diag().max()
  
    return E, L[:q], L[q], beta
