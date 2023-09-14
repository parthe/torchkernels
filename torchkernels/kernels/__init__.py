from torch.func import grad


class Kernel:
    def __init__(self):
        self._matrix = None
        pass

    def __call__(self, value, save=False):
        if save: 
            self.matrix = value
        else:
            return value

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @matrix.deleter
    def matrix(self):
        del self._matrix

    def grad1(self):
        return grad(self.__call__)

