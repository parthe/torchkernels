import torch
from torchkernels.kernels.radial import laplacian
from torchkernels.feature_maps import LaplacianORF

n = 10 # number of samples
input_dim = 10 # input dimension
num_features = int(3e4) # number of random features

is_cuda = torch.cuda.is_available()
DEV = torch.device("cuda") if is_cuda else torch.device("cpu")

X = torch.randn(n, input_dim, device=DEV)

kernel_matrix_exact = laplacian(X, X, length_scale=1.)
kernel_matrix_exact_norm = torch.linalg.matrix_norm(kernel_matrix_exact, 'fro')

feature_map = LaplacianORF(input_dim=input_dim, num_features=num_features, device=DEV)
Phi = feature_map(X)
kernel_matrix_approx = Phi @ Phi.T
kernel_matrix_delta = kernel_matrix_exact - kernel_matrix_approx
kernel_matrix_delta_norm = torch.linalg.matrix_norm(kernel_matrix_delta, 'fro')
print(f"Approximation error in Frobenius norm: {kernel_matrix_delta_norm / kernel_matrix_exact_norm}")
