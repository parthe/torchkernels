from torch.func import vmap, grad

def vectorize(kernel):
  """
  Takes a method `kernel` from Rd x Rd -> R
  """
  return vmap(
    vmap(kernel, in_dims=(None, 0)), 
    in_dims=(0, None))
