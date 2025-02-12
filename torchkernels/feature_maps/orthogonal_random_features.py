from .__init__ import CMS_sampling
import torch
import scipy.stats as stats
import numpy as np
import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ORF_w(p_feat, d_dim, Kernel='Laplace', length_scale=1., nu=None, alpha=None):
        """
    Generate Q and Rad_dist for ORF feature map
    
    Arguments
    ----------
    p_feat : int
        Number of random features
    d_dim : int
        dimension of underlying problem 
    Kernel : str
        Kernel type, options are ['Laplace', 'Gauss', 'Matern', 'ExpPower']
    length_scale : float
        length_scale for the kernel
    nu : float
        shape parameter for Matern kernel, ignored if kernel is not Matern
    alpha : float
        stability parameter for ExpPower kernel, ignored if kernel is not ExpPower
    
    Returns
    -------
    Q : torch.tensor, shape (D, d_dim)
        Orthogonal transform
    Rad_dist : torch.tensor, shape (D,)
        Radial distribution
    """
        
        Q_arr = []
        
        if Kernel == "Laplace":
            Rad_dist = np.sqrt(stats.betaprime.rvs(d_dim/2,1/2, size=p_feat))/length_scale
        elif Kernel =="Gauss":
            Rad_dist = stats.chi.rvs(d_dim,size=p_feat)/length_scale
        elif Kernel == "Matern":
            Rad_dist = np.sqrt(stats.betaprime.rvs(d_dim/2, nu, size=p_feat))*np.sqrt(2*nu)/length_scale
        elif Kernel == "ExpPower":
            Rad_dist = radial_CMS(p_feat=p_feat, d_dim=d_dim, alpha=alpha, length_scale=length_scale)
        for _ in range(int(np.ceil(p_feat/d_dim))):
            Q = stats.ortho_group.rvs(dim=d_dim)
            Q_arr.append(Q)
        Q = np.concatenate(Q_arr, axis=0)
        del Q_arr
        return torch.from_numpy((Q[:p_feat]).T).to(torch.float32), torch.from_numpy(Rad_dist).to(dtype=torch.float32)
    
def radial_CMS(p_feat, d_dim, alpha, length_scale=1.):
        """
    Arguments
    ----------
    p_feat : int
        Number of random features
    d_dim : int
        dimension of underlying problem
    alpha : float
        stability parameter for ExpPower kernel, ignored if kernel is not ExpPower
    length_scale : float
        length_scale for the kernel

    Returns
    -------
    x : torch.tensor, shape (p_feat,)
        Radial distribution
    """
    # Generate radial CMS given alpha, bw (bandwidth)
        x = CMS_sampling(p=p_feat, alpha=alpha, length_scale=length_scale)
        y = stats.chi.rvs(d_dim, size=p_feat)
        return x*y

class Orthogonal_Random_Features:
    def __init__(self, p_feat, d_dim, kernel="Laplace", Rf_bias:bool=False, length_scale=1., nu=None, alpha=None):
        assert kernel in ["Laplace", "Gauss", "Matern", "ExpPower"]
        if kernel == "Matern": assert nu is not None
        if kernel == "ExpPower": 
            assert alpha is not None
            assert 0 < alpha <= 2
            if alpha==1: raise NotImplementedError("alpha = 1 is Laplace Kernel use that instead")
            if alpha==2: raise NotImplementedError("alpha = 2 is Gaussian Kernel use that instead")

        
        self.p_feat = p_feat
        self.d_dim = d_dim
        self.kernel = kernel
        self.length_scale = length_scale
        self.nu = nu
        self.alpha = alpha
        self.Rf_bias = Rf_bias
        if not Rf_bias: p_feat = p_feat // 2
        self.Q, self.Rad_dist = ORF_w(p_feat=p_feat, d_dim=self.d_dim, Kernel=self.kernel, length_scale=self.length_scale, nu=self.nu, alpha=self.alpha)
        self.b = ((torch.rand(self.p_feat) * math.pi * 2).to(DEVICE))
    
    def __call__(self, X_):
        Q = self.Q.to(DEVICE)  # On GPU
        R_dist = self.Rad_dist.to(DEVICE)  # On GPU
        c1 = (torch.sqrt(torch.tensor(2 / self.p_feat)).to(DEVICE))  # On GPU
        
        torch.cuda.empty_cache()
        X_ = X_.to(DEVICE)  # On GPU
        if self.Rf_bias:
            return c1 * ((torch.mm(X_, Q)*R_dist) + self.b).cos()
        else:
            return c1 * torch.cat([(torch.mm(X_, Q)*R_dist).cos(), (torch.mm(X_, Q)*R_dist).sin()], dim=-1)

def create_ORF(Q, R_dist, X_, kernel='Laplace', Rf_bias:bool=False, length_scale:float=1., nu=None, alpha=None) -> torch.tensor:
    """
    Generate random features using Orthogonal Random Features (ORF) method.

    Parameters
    ----------
    Q : torch.tensor
        Stacked up orthogonal matrix.
    R_dist : torch.tensor
        Radial distribution vector.
    X_ : torch.tensor
        Input data tensor.
    kernel : str, optional
        Type of kernel to use; options are ['Laplace', 'Gauss', 'Matern', 'ExpPower'].
        Defaults to 'Laplace'.
    Rf_bias : bool, optional
        Whether to include a bias term in the random features. Defaults to False.
    length_scale : float, optional
        Length scale for the kernel. Defaults to 1.
    nu : float, optional
        Shape parameter for the Matern kernel, required if kernel is 'Matern'.
    alpha : float, optional
        Stability parameter for the ExpPower kernel, required if kernel is 'ExpPower'.

    Returns
    -------
    torch.tensor
        The generated random features.
    """

    assert kernel in ["Laplace", "Gauss", "Matern", "ExpPower"]
    if kernel == "Matern": assert nu is not None
    if kernel == "ExpPower": 
        assert alpha is not None
        assert 0 < alpha <= 2
        if alpha==1: raise NotImplementedError("alpha = 1 is Laplace Kernel use that instead")
        if alpha==2: raise NotImplementedError("alpha = 2 is Gaussian Kernel use that instead")

    if not Rf_bias:
        p_feat = p_feat // 2
    
    d_dim = X_.shape[1]
    # Q, R_dist = ORF_w(p_feat=p_feat, d_dim=d_dim, Kernel=kernel, length_scale=length_scale, nu=nu, alpha=alpha)
    Q = Q.to(DEVICE)  # On GPU
    R_dist = R_dist.to(DEVICE)  # On GPU
    b = ((torch.rand(p_feat) * torch.pi * 2).to(DEVICE))  # On GPU
    c1 = (torch.sqrt(torch.tensor(2 / p_feat)).to(DEVICE))  # On GPU
    
    torch.cuda.empty_cache()
    X_ = X_.to(DEVICE)  # On GPU
    if Rf_bias:
        return c1 * ((torch.mm(X_, Q)*R_dist) + b).cos()
    else:
        return c1 * torch.cat([(torch.mm(X_, Q)*R_dist).cos(), (torch.mm(X_, Q)*R_dist).sin()], dim=-1)