import functools
from time import time

from pytictoc import TicToc
from torch.func import vmap

timer = TicToc()


def vectorize(kernel_fn):
    """
  Vectorizes a bivariate method 
  If `fn` maps inputs (d1,) and (d2,) --> (d3,)
  `vectorize(fn)` maps inputs (n, d1) and (p, d2) --> (n, p, d3)
  """
    return functools.wraps(
        vmap(
            vmap(kernel_fn, in_dims=(None, 0)),
            in_dims=(0, None)
        ),
        kernel_fn
    )


def timeit(func):
    # This function shows the execution time of the function object passed
    @functools.wraps(func)
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func
