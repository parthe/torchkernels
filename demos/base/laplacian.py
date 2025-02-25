import torch
from torchkernels.kernels.radial import laplacian, LaplacianKernel

n = 300 # number of samples
p = 200 # number of centers
d = 100  # dimensions

is_cuda = torch.cuda.is_available()
DEV = torch.device("cuda") if is_cuda else torch.device("cpu")    

X = torch.randn(n, d, device=DEV)
Z = torch.randn(p, d, device=DEV)

kernel_matrix1 = laplacian(X, Z, bandwidth=1.)

K = LaplacianKernel(bandwidth=1.)
kernel_matrix2 = K(X, Z)

torch.testing.assert_close(kernel_matrix1, kernel_matrix2, msg='Laplacian test failed')
print('Laplacian test complete!')
