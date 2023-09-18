import torch


def rp_cholesky_sampler(K, X, subsample_size=10, alg='rp'):
    n = X.shape[0]
    diags = K(X).diag() #(n,)
    F = torch.zeros(subsample_size, n)
    arr_idx = []
    
    for i in range(subsample_size):
        if alg == 'rp':
            idx = torch.multinomial(diags/diags.sum(), 1)
        elif alg == 'greedy':
            idx = diags.argmax().unsqueeze(0)
        else:
            raise RuntimeError(f"Algorithm '{alg}' not recognized")
        
        arr_idx.append(idx)
        F[i] = (K(X[idx], X) - F[:i, idx].T @ F[:i])/diags[idx].sqrt()
        diags -= F[i]**2
        diags = diags.clip(min=0)

    return F, X[torch.cat(arr_idx)], torch.cat(arr_idx)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from torchkernels.kernels.radial import LaplacianKernel
    
    K = LaplacianKernel(bandwidth=1.)

    n, d, k = 100, 2, 20
    X = torch.randn(n, d)
    
    Fn, Xn, idxn = rp_cholesky_sampler(K, X, k, alg='rp')
    Fr, Xr, idxr = rp_cholesky_sampler(K, X, k, alg='greedy')
    plt.scatter(X[:,0], X[:,1], marker='o', c='gray', s=10, label='data')
    plt.scatter(Xn[:,0], Xn[:,1], marker='*', c='green', s=100, label='RPC subsamples')
    plt.scatter(Xr[:,0], Xr[:,1], marker='.', c='red', s=100, label='greedy subsamples')
    plt.legend()
    plt.title('RP Cholesky')
    plt.show()