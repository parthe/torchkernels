from .primitives import norm, inner_product

def normalized_inner_product(func, x, z, M=None):
    """
        K(x,z) = norm(x) * norm(z) * func(<x_, z_>) where x_ = x/norm(x) and z_ = z/norm(z)
    """
    x_norm = norm(x, M=M)
    z_norm = norm(z, M=M)
    xz = inner_product(x/x_norm.view(-1,1), z/z_norm.view(-1,1), M=M)
    return x_norm.view(-1,1) * func(xz) * z_norm     
