from torch import nn

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
