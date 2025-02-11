import torch
import torch.distributions as dist
def CMS_sampling(p, alpha, length_scale=1.):
    r"""
    Generate radial CMS given alpha, length_scale and dimension d. Samples generated are from $S(\alpha/2, 1, $2 \gamma^2 (cos(\pi \alpha/2))^(2/\alpha)$, 0)
    
    Parameters:
        p (int): Number of samples
        alpha (float): alpha for the multivariable alpha-stable distribution
        length_scale (float): length_scale for the distribution, default is 1.
    
    Returns:
        x (torch.tensor), shape (n,)): generated CMS samples
    """
    PI = torch.tensor(torch.pi)
    gamma_2 = 1/(length_scale**2)
    sigma = 2 * gamma_2* ((PI*alpha/4).cos())**(2/alpha)
    if not 0. < alpha < 2.:
        raise ValueError("Alpha must be between 0 and 2, both excluded")
    v = torch.rand(p)*PI - PI/2
    w = dist.exponential.Exponential(torch.tensor(1.)).sample((p,))
    
    alpha = alpha/2
    if alpha == 1.:
        raise NotImplementedError("alpha = 1 is Laplace Kernel use that instead")
    elif alpha == 2.:
        raise NotImplementedError("alpha = 2 is Gaussian Kernel use that instead")
    else:
        arg1 = PI/2 * alpha
        b_ab = PI/2
        s_ab = torch.pow(arg1.cos(),-1/alpha)
        arg2 = alpha * (v + b_ab)
        n1 = arg2.sin()
        d1 = torch.pow(v.cos(),1/alpha)
        n2 = (v-arg2).cos()
        x = sigma * s_ab * (n1/d1) * torch.pow(n2/w, (1-alpha)/alpha)
        return x

# if __name__ == "__main__":
#     print(CMS_sampling(p=100, alpha=0.7, length_scale=1.))