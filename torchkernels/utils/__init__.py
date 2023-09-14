from torch.func import vmap, grad
import functools


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
    
@vectorize
def grad1(kernel):
    return grad(kernel)
