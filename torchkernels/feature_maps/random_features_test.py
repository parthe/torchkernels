from ..kernels.radial import laplacian, gaussian, exp_power, matern
from .orthogonal_random_features import create_ORF
# from .random_fourier_features import create_RFF
import torch
import matplotlib.pyplot as plt
import numpy as np

print("Imports done")
def test_kernel_approximation(d=32, D=int(1e5), bandwidth=1., alpha=0.5, nu=1.5, App_type="ORF", Kernel = 'ExpPower'):
    assert Kernel in ["Laplace", "Gauss", "Matern", "ExpPower"]
    K_func = {"Laplace": laplacian, "Gauss": gaussian, "Matern": matern, "ExpPower": exp_power}
    N = int(50)
    x1_x2_norm = np.linspace(0, 5, N)
    K_mat_exact = []
    K_mat_approx = []
    for i, x_ in enumerate(x1_x2_norm):
        x1 = torch.randn(d)
        x1_x2_u = torch.randn(d)
        x1_x2 = x1_x2_u/np.linalg.norm(x1_x2_u)*x_
        x2 = x1 + x1_x2
        ORF_Phi_x1 = create_ORF(p_feat=D, X_=x1, kernel=Kernel, Rf_bias=False, length_scale=bandwidth, nu=nu, alpha=alpha)
        ORF_Phi_x2 = create_ORF(p_feat=D, X_=x2, kernel=Kernel, Rf_bias=False, length_scale=bandwidth, nu=nu, alpha=alpha)
        K_mat_approx.append(torch.dot(ORF_Phi_x1, ORF_Phi_x2).item())
        x1 = x1.reshape(1,-1)
        x2 = x2.reshape(1,-1)
        K_mat_exact.append(K_func[Kernel](x1_x2, bandwidth=bandwidth, nu=nu, alpha=alpha))
    plt.figure(figsize=(6, 6))
    plt.plot(x1_x2_norm, K_mat_exact, label=f'Laplace kernel (bandwidth={bandwidth})')
    plt.scatter(x1_x2_norm, K_mat_approx, label = 'RFF Estimate', alpha=0.5, color='red')
    plt.ylim(1e-4, 1.5)
    plt.xlabel("||x-z||")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.yscale('log')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    test_kernel_approximation()