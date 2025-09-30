from ..linalg import euclidean
from .__init__ import Kernel
import torch
from sklearn.gaussian_process.kernels import Matern


class RadialKernel(Kernel):
    '''
        K(x,z)=phi(-\norm{x-z}_M / length_scale) where phi is specified in subclass
    '''
    def __init__(self, fn=None, length_scale=1., squared=True, in_place=True):
        super().__init__()
        self.fn = fn
        self.squared = squared
        assert length_scale > 0, "argument `length_scale` must be positive."
        self.length_scale = 2 * length_scale**2 if squared else length_scale
        self.in_place = in_place

    def __call__(self, samples, centers=None, M=None, **kwargs):
        if centers is None: 
            centers = samples
        matrix = euclidean(samples, centers, squared=self.squared, M=M, in_place=self.in_place)
        if self.in_place:
            matrix.div_(-self.length_scale)
            # if matrix.device=='cuda': raise NotImplementedError("Currently `torch.Tensor.apply_` is not supported on CUDA")
            matrix = self.fn(matrix)
            return matrix
        else:
            matrix.div(-self.length_scale)
            return self.fn(matrix)

class LaplacianKernel(RadialKernel):
    def __init__(self, length_scale=1.):
        super().__init__(fn=lambda x: torch.exp(x), length_scale=length_scale, squared=False)
    
class GaussianKernel(RadialKernel):
    def __init__(self, length_scale=1.):
        super().__init__(fn=lambda x: torch.exp(x), length_scale=length_scale, squared=True)
    
class ExponentialPowerKernel(RadialKernel):
    def __init__(self, length_scale=1., power=1.):
        super().__init__(fn=lambda x: torch.exp(torch.pow(x, power/2)), length_scale=length_scale, squared=True)

def laplacian(samples, centers=None, length_scale=1., M=None, in_place=True):
    '''
        K(x,z)=exp(-\norm{x-z}_M / length_scale)
    '''
    assert length_scale > 0
    if centers is None: centers = samples
    kernel_mat = euclidean(samples, centers, squared=False, M=M, in_place=in_place)
    if in_place:
        kernel_mat.div_(-length_scale)
        kernel_mat.exp_()
        return kernel_mat
    else:
        kernel_mat = kernel_mat.div(-length_scale)
        return kernel_mat.exp()

def gaussian(samples, centers=None, length_scale=1., M=None, in_place=True):
    '''
        K(x,z)=exp(-\norm{x-z}_M^2 / 2/length_scale^2)
    '''
    assert length_scale > 0
    if centers is None: centers = samples
    kernel_mat = euclidean(samples, centers, squared=True, M=M, in_place=in_place)
    if in_place:
        kernel_mat.div_(-2 * length_scale ** 2)
        kernel_mat.exp_()
        return kernel_mat
    else:
        kernel_mat = kernel_mat.div(-2 * length_scale ** 2)
        return kernel_mat.exp()

def exp_power(samples, centers=None, length_scale=1., alpha=1., M=None, in_place=True):
    '''
        K(x,z)=exp(-(\norm{x-z}_M/length_scale)^\alpha)
    '''
    assert length_scale > 0
    if centers is None: centers = samples
    kernel_mat = euclidean(samples, centers, squared=True, M=M, in_place=in_place)
    if in_place:
        kernel_mat.pow_(alpha /2.)
        kernel_mat.div_(-(length_scale**alpha))
        kernel_mat.exp_()
        return kernel_mat
    else:
        kernel_mat = kernel_mat.pow(alpha /2.)
        kernel_mat = kernel_mat.div(-(length_scale**alpha))
        return kernel_mat.exp()

    
def matern(samples, centers=None, length_scale=1., nu=1., M=None):
    if M is not None: raise NotImplementedError
    X = samples.cpu().numpy()
    Z = centers.cpu().numpy()
    kernel = Matern(length_scale=length_scale, nu=nu)
    K_mat = kernel(X, Z)
    del X, Z
    return torch.from_numpy(K_mat)#.to(torch.float32)


# Aliases
exponential_power = exp_power
rbf = gaussian
laplace = laplacian
