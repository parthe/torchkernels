from .primitives.linalg import euclidean

def laplacian(samples, centers, bandwidth=1., M=None):
    '''
        K(x,z)=exp(-\norm{x-z}_M / bandwidth)
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers, squared=False, M=M)
    kernel_mat.clamp_(min=0)
    gamma = 1. / bandwidth
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def gaussian(samples, centers, bandwidth=1., M=None):
    '''
        K(x,z)=exp(-\norm{x-z}_M^2 / 2/bandwidth^2)
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers, squared=True, M=M)
    kernel_mat.clamp_(min=0)
    gamma = 1. / (2 * bandwidth ** 2)
    kernel_mat.mul_(-gamma)
    kernel_mat.exp_()
    return kernel_mat

def exp_power(samples, centers, bandwidth=1., power=1., M=None):
    '''
        K(x,z)=exp(-\norm{x-z}_M^\gamma / bandwidth)
    '''
    assert bandwidth > 0
    kernel_mat = euclidean(samples, centers, squared=True, M=M)
    kernel_mat.pow_(gamma / 2.)
    kernel_mat.mul_(-1. / bandwidth)
    kernel_mat.exp_()
    return kernel_mat

# Aliases
exponential_power = exp_power
dispersal = exp_power
rbf = gaussian
laplace = laplacian
