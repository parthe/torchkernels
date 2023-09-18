import torch, scipy

def top_eigensystem(K, X, q, method='scipy.linalg.eigh'):
    assert method in {"scipy.linalg.eigh", "torch.lobpcg"}
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
        beta: max{i} of K(xi, xi) - \sum_j=1^q (L[i]-lqp1) psi_j(xi)**2
    """
  
    n = X.shape[0]
    scaled_kmat = K(X, X)/n
    if method == "scipy.linalg.eigh":
      L, E = scipy.linalg.eigh(scaled_kmat.cpu().numpy(), subset_by_index=[n-q-1,n-1])
      L, E = torch.from_numpy(L).to(scaled_kmat.device), torch.from_numpy(E).to(scaled_kmat.device)
    elif method == "torch.lobpcg":
      L, E = torch.lobpcg(scaled_kmat, q+1)
    beta = n * (scaled_kmat.diag() - (E[:,:q].pow(2)*(L[:q]-L[q])).sum(-1)).max()
  
    return E[:,:q], L[:q], L[q], beta

def nystrom_extension(K, X, Xs, E):
    """
        Extend eigenvectors
    """
    E_ = K(X, Xs) @ E
    return E_/E_.norm(dim=0, keepdim=True)
