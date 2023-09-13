from torch.func import vmap, grad

def vectorize(fn):
  """
  Vectorizes a bivariate method 
  If `fn` maps inputs (d1,) and (d2,) --> (d3,)
  `vectorize(fn)` maps inputs (n, d1) and (p, d2) --> (n, p, d3)
  """
  return vmap(
    vmap(kernel, in_dims=(None, 0)), 
    in_dims=(0, None))
