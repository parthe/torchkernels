from .orf import ORF
from .rff import RFF
import scipy.stats as stats
import numpy as np
import torch
from .utils import CMS_sampling


class ExpPowerORF(ORF):
	def __init__(self, *args, alpha:float=None, **kwargs):
		"""
		Parameters
		----------
		input_dim : int
			input dimension of the data.
		num_features : int
			number of random features to generate.
		length_scale : float
			kernel length scale, defaults to 1.
		shape_matrix : torch.Tensor
			shape matrix for random features, defaults to None. shape matrix entered must be symmetric, positive definite and of dimension d x d where d is input dimension.
		bias_term : bool
			whether to include a bias term in the random features, defaults to False.
		device : str
			which device to use, can be 'cpu' or 'cuda', defaults to None which means use cuda if available.
		float_type : torch.dtype
			float type to use, defaults to torch.float64.
		alpha : float
			stability parameter for the ExpPower kernel, must be between 0 and 2, both not included. Defaults to None.
		"""
		assert alpha is not None
		assert alpha > 0 and alpha < 2
		if alpha==1: raise NotImplementedError("alpha = 1 is Laplacian kernel. use torchkernels.feature_maps.LaplacianORF")
		if alpha==2: raise NotImplementedError("alpha = 2 is Gaussian kernel. use torchkernels.feature_maps.GaussianORF")
		self.alpha = alpha
		super().__init__(*args, **kwargs)

	def set_S(self, S=None):
		if S is not None:
			self.S=S
		else:
			CMS_samples = CMS_sampling(p=self._num_features, alpha=self.alpha, length_scale=self.length_scale)
			Chi_samples = stats.chi.rvs(self.input_dim, size=self._num_features)
			self.S = torch.from_numpy(np.sqrt(CMS_samples)*Chi_samples).to(self.float_type).to(self.device) 


class ExpPowerRFF(RFF):
	def __init__(self, *args, alpha:float=None, **kwargs):
		"""Initialize an instance of the RFF class.
		
		Parameters
		----------
		input_dim : int
			input dimension of the data.
		num_features : int
			number of random features to generate.
		length_scale : float
			kernel length scale, defaults to 1.
		shape_matrix : torch.Tensor
			shape matrix for random features, defaults to None. shape matrix entered must be symmetric, positive definite and of dimension d x d where d is input dimension.
		bias_term : bool
			whether to include a bias term in the random features, defaults to False.
		device : str
			which device to use, can be 'cpu' or 'cuda', defaults to None which means use cuda if available.
		float_type : torch.dtype
			float type to use, defaults to torch.float64.
		alpha : float
			stability parameter for the ExpPower kernel, must be between 0 and 2, both not included. Defaults to None.
		"""
		assert alpha is not None
		assert alpha > 0 and alpha < 2
		if alpha==1: raise NotImplementedError("alpha = 1 is Laplace kernel. use torchkernels.feature_maps.LaplacianRFF")
		if alpha==2: raise NotImplementedError("alpha = 2 is Gaussian kernel. use torchkernels.feature_maps.GaussianRFF")
		self.alpha = alpha
		super().__init__(*args, **kwargs)

	def set_W2(self, W2=None):
		if W2 is not None:
			self.W2=W2 
		else:
			self.W2 = torch.from_numpy(
				np.sqrt(
					CMS_sampling(p=self._num_features,  alpha=self.alpha, length_scale=1.)
				)).to(self.float_type).to(self.device)

	def apply_W2(self, XW1):
		if self.bias_term:
			return self.c1 * ((XW1*self.W2) + self._bias).cos()
		else:
			return self.c1 * torch.cat([(XW1*self.W2).cos(), (XW1*self.W2).sin()], dim=-1)
