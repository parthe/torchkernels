from .radial import laplacian, gaussian, exp_power
from torch import nn

# Aliases
exponential_power = exp_power
dispersal = exp_power
rbf = gaussian
laplace = laplacian


class Kernel(nn.Module):
    def __init__(self):
        self._matrix = None
        pass
      
    def forward(self, x, z=None, save=False):
        pass

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @matrix.deleter
    def matrix(self):
        del self._matrix
