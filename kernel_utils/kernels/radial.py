from ..linalg import euclidean
from torch import nn

class RadialKernel(nn.Module):
    '''
        K(x,z)=phi(-\norm{x-z}_M / bandwidth) where phi is specified in subclass
    '''
    def __init__(self, fn=None, bandwidth=1., squared=True):
        self.fn = fn
        self.squared = squared
        assert bandwidth > 0, "argument `bandwidth` must be positive."
        self.bandwidth = 2 * bandwidth**2 if squared else bandwidth

    def forward(self, samples, centers=None, M=None):
        if centers is None: 
            centers = samples
        return self.fn(euclidean(samples, centers, squared=self.squared, M=M).div_(-self.bandwidth))

class LaplacianKernel(RadialKernel):
    def __init__(self, bandwidth=1.):
        def phi(k): k.exp_()
        super().__init__(fn=phi, squared=False, bandwidth=bandwidth)
    
class GaussianKernel(RadialKernel):
    def __init__(self, bandwidth=1.):
        def phi(k): k.exp_()
        super().__init__(fn=phi, squared=True, bandwidth=bandwidth)
    
class ExpPowerKernel(RadialKernel):
    def __init__(self, power=1.,bandwidth=1.):
        def phi(k): k.pow_(power/2).exp_()
        super().__init__(fn=phi, squared=True, bandwidth=bandwidth)

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
