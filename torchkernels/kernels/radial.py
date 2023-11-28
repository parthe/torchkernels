from ..linalg import euclidean
from .__init__ import Kernel


class RadialKernel(Kernel):
    '''
        K(x,z)=phi(-\norm{x-z}_M / bandwidth) where phi is specified in subclass
    '''
    def __init__(self, fn=None, bandwidth=1., squared=True):
        super().__init__()
        self.fn = fn
        self.squared = squared
        assert bandwidth > 0, "argument `bandwidth` must be positive."
        self.bandwidth = 2 * bandwidth**2 if squared else bandwidth

    def __call__(self, samples, centers=None, M=None, **kwargs):
        if centers is None: 
            centers = samples
        matrix = euclidean(samples, centers, squared=self.squared, M=M)
        matrix.div_(-self.bandwidth)
        if matrix.device=='cuda': raise NotImplementedError("Currently `torch.Tensor.apply_` is not supported on CUDA")
        matrix.apply_(self.fn)
        return matrix

class LaplacianKernel(RadialKernel):
    def __init__(self, bandwidth=1.):
        super().__init__(fn=lambda x: torch.exp(x), bandwidth=bandwidth, squared=False)
    
class GaussianKernel(RadialKernel):
    def __init__(self, bandwidth=1.):
        super().__init__(fn=lambda x: torch.exp(x), bandwidth=bandwidth, squared=True)
    
class ExponentialPowerKernel(RadialKernel):
    def __init__(self, bandwidth=1.):
        super().__init__(fn=lambda x: torch.exp(torch.pow(power/2)), bandwidth=bandwidth, squared=True)

def laplacian(samples, centers=None, bandwidth=1., M=None):
    '''
        K(x,z)=exp(-\norm{x-z}_M / bandwidth)
    '''
    assert bandwidth > 0
    if centers is None: centers = samples
    kernel_mat = euclidean(samples, centers, squared=False, M=M)
    kernel_mat.div_(-bandwidth)
    kernel_mat.exp_()
    return kernel_mat

def gaussian(samples, centers=None, bandwidth=1., M=None):
    '''
        K(x,z)=exp(-\norm{x-z}_M^2 / 2/bandwidth^2)
    '''
    assert bandwidth > 0
    if centers is None: centers = samples
    kernel_mat = euclidean(samples, centers, squared=True, M=M)
    kernel_mat.div_(-2 * bandwidth ** 2)
    kernel_mat.exp_()
    return kernel_mat

def exp_power(samples, centers=None, bandwidth=1., power=1., M=None):
    '''
        K(x,z)=exp(-\norm{x-z}_M^\gamma / bandwidth)
    '''
    assert bandwidth > 0
    if centers is None: centers = samples
    kernel_mat = euclidean(samples, centers, squared=True, M=M)
    kernel_mat.pow_(power / 2.)
    kernel_mat.div_(-bandwidth)
    kernel_mat.exp_()
    return kernel_mat


# Aliases
exponential_power = exp_power
dispersal = exp_power
rbf = gaussian
laplace = laplacian
