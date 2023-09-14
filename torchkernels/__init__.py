__version__ = 0.1

from torchkernels.utils import vectorize


@vectorize
def grad1(kernel_fn):
    return grad(kernel_fn)
