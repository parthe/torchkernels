from .__init__ import CMS_sampling
import torch
from torch.distributions.chi2 import Chi2
import scipy.stats as stats
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def RFF_w(p_feat, d_dim, Kernel="Laplace", length_scale=1., nu=None, alpha=None, file_handle=None):
    """
    Generate 1 weight matrices Random Fourier Features (RFF) method with specified kernel. 
    For the Laplace and the Matern kernel W = W1/W2.
    For the ExpPower kernel W = W1*W2.
    For the Gaussian kernel W2 is just all ones.

    Parameters
    ----------
    p_feat : int    
        Number of random features.
    d_dim : int
        dimension of underlying dataset 
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
    W1 : torch.tensor, shape (d_dim, p_feat)
        W1 for RFF
    W2 : torch.tensor, shape (p_feat,)
        W2 for RFF
        
    """
    assert Kernel in ["Laplace", "Gauss", "Matern", "ExpPower"]
    if Kernel == "Laplace":
        return torch.randn(d_dim,p_feat)*length_scale , torch.randn(p_feat)
    elif Kernel =="Gauss":
        return torch.randn(d_dim,p_feat)*length_scale , torch.ones(p_feat)
    elif Kernel == "Matern":
        df=2*nu
        chi2_dist = Chi2(df=df)
        chi2_samples = chi2_dist.sample((p_feat,))/(2*nu)
        return torch.randn(d_dim,p_feat)/length_scale , torch.sqrt(chi2_samples)
    elif Kernel == "ExpPower":
        return torch.randn(d_dim,p_feat), torch.sqrt(torch.from_numpy(CMS_sampling(p=p_feat,  alpha=alpha, length_scale=length_scale)))
    
class Random_Fourier_Features:
    def __init__(self, p_feat, d_dim, kernel="Laplace", Rf_bias:bool=False, length_scale=1., nu=None, alpha=None):
        """
        Initializes the class for creating Random Fourier Features
        
        Parameters
        ----------
        p_feat : int    
            Number of random features.
        d_dim : int
            dimension of underlying dataset 
        kernel : str
            Kernel type, options are ['Laplace', 'Gauss', 'Matern', 'ExpPower']
        Rf_bias : bool, optional
            Whether to include a bias term in the random features. Defaults to False.
        length_scale : float, >0
            length_scale for the kernel, strictly greater than 0.
        nu : float, optional
            shape parameter for Matern kernel, required if kernel is 'Matern'.
        alpha : float, optional
            stability parameter for ExpPower kernel, required if kernel is 'ExpPower'.
        """
        
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
        self.W1, self.W2 = RFF_w(p_feat=p_feat, d_dim=self.d_dim, Kernel=self.kernel, length_scale=self.length_scale, nu=self.nu, alpha=self.alpha)
    def __call__(self, X_):
        W1 = self.W1.to(DEVICE)  # On GPU
        W2 = self.W2.to(DEVICE)  # On GPU
        b = ((torch.rand(self.p_feat) * torch.pi * 2).to(DEVICE))  # On GPU
        c1 = (torch.sqrt(torch.tensor(2 / self.p_feat)).to(DEVICE))  # On GPU
        X_ = X_.to(DEVICE)  # On GPU
        if self.Rf_bias:
            if self.kernel in ['Laplace', 'Matern']:
                return c1 * ((torch.mm(X_, W1)/W2) + b).cos()
            elif self.kernel in ['Gauss', 'ExpPower']:
                return c1 * ((torch.mm(X_, W1)*W2) + b).cos()
        else:
            if self.kernel in ['Laplace', 'Matern']:
                return c1 * torch.cat([(torch.mm(X_, W1)/W2).cos(), (torch.mm(X_, W1)/W2).sin()], dim=-1)
            elif self.kernel in ['Gauss', 'ExpPower']:
                return c1 * torch.cat([(torch.mm(X_, W1)*W2).cos(), (torch.mm(X_, W1)*W2).sin()], dim=-1)

def create_RFF(W1, W2, X_, kernel='Laplace', Rf_bias:bool=False, length_scale:float=1., nu=None, alpha=None) -> torch.tensor:
    """
    Generate random features using Orthogonal Random Features (ORF) method.

    Parameters
    ----------
    p_feat : int    
        Number of random features.
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
    W1, W2 = RFF_w(D=p_feat, d=d_dim, Kernel=kernel, length_scale=length_scale, nu=nu, alpha=alpha)
    W1 = W1.to(DEVICE)  # On GPU
    W2 = W2.to(DEVICE)  # On GPU
    b = ((torch.rand(p_feat) * torch.pi * 2).to(DEVICE))  # On GPU
    c1 = (torch.sqrt(torch.tensor(2 / p_feat)).to(DEVICE))  # On GPU
    
    torch.cuda.empty_cache()
    X_ = X_.to(DEVICE)  # On GPU
    if Rf_bias:
        if kernel in ['Laplace', 'Matern']:
            return c1 * ((torch.mm(X_, W1)/W2) + b).cos()
        elif kernel in ['Gauss', 'ExpPower']:
            return c1 * ((torch.mm(X_, W1)*W2) + b).cos()
    else:
        if kernel in ['Laplace', 'Matern']:
            return c1 * torch.cat([(torch.mm(X_, W1)/W2).cos(), (torch.mm(X_, W1)/W2).sin()], dim=-1)
        elif kernel in ['Gauss', 'ExpPower']:
            return c1 * torch.cat([(torch.mm(X_, W1)*W2).cos(), (torch.mm(X_, W1)*W2).sin()], dim=-1)