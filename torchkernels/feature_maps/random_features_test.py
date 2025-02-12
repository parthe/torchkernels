from ..kernels.radial import laplacian, gaussian, exp_power, matern
from .orthogonal_random_features import Orthogonal_Random_Features
from .random_fourier_features import Random_Fourier_Features
import torch
import matplotlib.pyplot as plt
import numpy as np

print("Imports done")
def test_kernel_approximation(d=10, D=int(1e3), length_scale=1., alpha=0.7, nu=1.5, App_type="ORF", Rf_bias:bool=False, kernel = 'ExpPower'):
    assert kernel in ["Laplace", "Gauss", "Matern", "ExpPower"]
    K_func = {"Laplace": laplacian, "Gauss": gaussian, "Matern": matern, "ExpPower": exp_power}
    N = int(50)
    x1_x2_norm = np.linspace(0, 5, N)
    K_mat_exact = []
    K_mat_approx = []
    if App_type == "ORF":
        RF_obj = Orthogonal_Random_Features(p_feat=D, d_dim=d, kernel=kernel, Rf_bias=Rf_bias, length_scale=length_scale, nu=nu, alpha=alpha)
    elif App_type == "RFF":
        RF_obj = Random_Fourier_Features(p_feat=D, d_dim=d, kernel=kernel, Rf_bias=Rf_bias, length_scale=length_scale, nu=nu, alpha=alpha)
    else:
        raise ValueError("App_type must be either 'ORF' or 'RFF'")

    for i, x_ in enumerate(x1_x2_norm):
        x1 = torch.randn(d)
        x1_x2_u = torch.randn(d)
        x1_x2 = x1_x2_u/np.linalg.norm(x1_x2_u)*x_
        x2 = x1 + x1_x2
        x1 = x1.reshape(1,-1)
        x2 = x2.reshape(1,-1)
        Phi_x1 = RF_obj(x1)
        Phi_x2 = RF_obj(x2)
        K_mat_approx.append((Phi_x1@Phi_x2.T)[0].item())
        if kernel == "ExpPower":
            K_mat_exact.append(K_func[kernel](x1, x2, length_scale=length_scale, alpha=alpha)[0].item())
        elif kernel == "Matern":
            K_mat_exact.append(K_func[kernel](x1, x2, length_scale=length_scale, nu=nu)[0].item())
        else:
            K_mat_exact.append(K_func[kernel](x1, x2, length_scale=length_scale)[0].item())
    plt.figure(figsize=(6, 6))
    plt.scatter(x1_x2_norm, K_mat_approx, label = f'{App_type} Estimate', alpha=0.5, color='red')
    plt.plot(x1_x2_norm, K_mat_exact, label=f'exact')
    plt.ylim(1e-4, 1.5)
    plt.xlabel("||x-z||")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    # plt.yscale('log')
    plt.legend()
    plt.show()

for kernel in ["Laplace", "Gauss", "Matern", "ExpPower"]:
    for app_type in ["ORF", "RFF"]:
        for rf_bias in [True, False]:
            print(kernel, app_type, rf_bias)
            test_kernel_approximation(d=32, D=int(1e5), length_scale=1.5, alpha=1.5, nu=1.5, App_type=app_type, Rf_bias=rf_bias, kernel = kernel)

# if __name__ == '__main__':
    # test_kernel_approximation()