import numpy as np

def CMS_sampling(p, alpha, length_scale=1., seed = None):
    r"""
    Generate radial CMS given alpha, length_scale and dimension d. Samples generated are from $S(\alpha/2, 1, $2 \gamma^2 (cos(\pi \alpha/2))^(2/\alpha)$, 0)
    
    Parameters:
        p (int): Number of samples
        alpha (float): alpha for the multivariable alpha-stable distribution
        length_scale (float): length_scale for the distribution, default is 1.
        seed (int): defaults to None. 
    
    Returns:
        x (torch.tensor), shape (n,)): generated CMS samples
    """
    PI = np.pi
    gamma_2 = 1/(float(length_scale))**2
    sigma = 2 * gamma_2* (np.cos(PI*alpha/4))**(2/alpha)
    if not 0. <= alpha <= 2.:
        raise ValueError("Alpha must be between 0 and 2")
    # generate random variables
    numpy_rng = np.random.default_rng(seed=seed)
    v = numpy_rng.uniform(-0.5 * PI, 0.5 * PI, p)
    w = numpy_rng.exponential(1, p)
    
    #alpha to be used in CMS is alpha/2 given alpha for multivariable stable distribution
    alpha = float(alpha)/2
    if alpha == 1.:
        raise NotImplementedError("alpha = 1 is Laplace Kernel use that instead")
    elif alpha == 2.:
        raise NotImplementedError("alpha = 2 is Gaussian Kernel use that instead")
    else:
        arg1 = 0.5 * PI * alpha
        b_ab = PI/2
        s_ab = np.cos(arg1)**(-1/alpha)
        arg2 = alpha * (v + b_ab)
        n1 = np.sin(arg2)
        d1 = np.cos(v)**(1/alpha)
        n2 = np.cos(v - arg2)
        x = sigma * s_ab * (n1/d1) * (n2/w)**((1-alpha)/alpha)
        return x

# if __name__ == "__main__":
#     print(CMS_sampling(p=100, alpha=0.7, length_scale=1.))