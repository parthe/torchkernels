from .orf import ORF
from .rff import RFF
import scipy.stats as stats
import numpy as np
import torch
from .utils import CMS_sampling


class ExpPowerORF(ORF):
	def __init__(self, *args, alpha:float=None, **kwargs):
		"""Initialize an instance of the ORF class.
		
		Parameters
		----------
		input_dim : int
			Input dimension of the data.
		num_features : int
			Number of random features to generate.
		length_scale : float
			Kernel length scale, defaults to 1.
		bias_term : bool
			Whether to include a bias term in the random features, defaults to False.
		device : str
			Which device to use, can be 'cpu' or 'cuda', defaults to None which means use cuda if available.
		alpha : float
			Stability parameter for the ExpPower kernel, must be between 0 and 2, both not included. Defaults to None.
		"""
		assert alpha is not None
		assert alpha > 0 and alpha < 2
		if alpha==1: raise NotImplementedError("alpha = 1 is Laplacian kernel use that instead")
		if alpha==2: raise NotImplementedError("alpha = 2 is Gaussian kernel use that instead")
		self.alpha = alpha
		super().__init__(*args, **kwargs)

	def set_S(self):
		CMS_samples = CMS_sampling(p=self.num_features, alpha=self.alpha, length_scale=self.length_scale)
		Chi_samples = stats.chi.rvs(self.input_dim, size=self.num_features)
		self.S = torch.from_numpy(np.sqrt(CMS_samples)*Chi_samples).to(self.device) 

	def test_set_S(self, S):
		self.S = S

class ExpPowerRFF(RFF):
	def __init__(self, *args, alpha:float=None, **kwargs):
		assert alpha is not None
		assert alpha > 0 and alpha < 2
		if alpha==1: raise NotImplementedError("alpha = 1 is Laplace Kernel use that instead")
		if alpha==2: raise NotImplementedError("alpha = 2 is Gaussian Kernel use that instead")
		self.alpha = alpha
		super().__init__(*args, **kwargs)

	def set_W2(self):
		self.W2 = torch.from_numpy(
			np.sqrt(
				CMS_sampling(p=self.num_features,  alpha=self.alpha, length_scale=1.)
			)).to(self.device)

	def test_set_W2(self, W2):
		self.W2 = W2

	def apply_W2(self, XW1):
		if self.bias_term:
			return self.c1 * ((XW1*self.W2) + self._bias).cos()
		else:
			return self.c1 * torch.cat([(XW1*self.W2).cos(), (XW1*self.W2).sin()], dim=-1)
