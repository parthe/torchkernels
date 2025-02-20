from .orf import ORF
from .rff import RFF
import scipy.stats as stats
from torch.distributions.chi2 import Chi2
import numpy as np
import torch


class MaternORF(ORF):
	def __init__(self, *args, nu:float=None, **kwargs):
		"""Initialize an instance of the ORF class.
		
		Parameters
		----------
		input_dim : int
			Input dimension of the data.
		num_features : int
			Number of random features to generate.
		length_scale : float
			Kernel length scale, defaults to 1.
		shape_matrix : torch.Tensor
			Shape matrix for the Matern kernel, defaults to None. shape matrix entered must be symmetric, positive definite and of dimension d x d where d is input dimension.
		bias_term : bool
			Whether to include a bias term in the random features, defaults to False.
		device : str
			Which device to use, can be 'cpu' or 'cuda', defaults to None which means use cuda if available.
		nu : float
			Smoothness parameter for the Matern kernel, must be greater than 0. Defaults to None.
		"""
		assert nu is not None
		assert nu>0
		self.nu = nu
		super().__init__(*args, **kwargs)


	def set_S(self, S=None):
		if S is not None:
			self.S=S
		else:
			self.S = torch.from_numpy(np.sqrt(stats.betaprime.rvs(self.input_dim/2, self.nu, size=self._num_features))
                             /self.length_scale*np.sqrt(2*self.nu)).to(self.device)


class MaternRFF(RFF):
	def __init__(self, *args, nu:float=None, **kwargs):
		"""Initialize an instance of the RFF class.
		
		Parameters
		----------
		input_dim : int
			Input dimension of the data.
		num_features : int
			Number of random features to generate.
		length_scale : float
			Kernel length scale, defaults to 1.
		shape_matrix : torch.Tensor
			Shape matrix for the Matern kernel, defaults to None. shape matrix entered must be symmetric, positive definite and of dimension d x d where d is input dimension.
		bias_term : bool
			Whether to include a bias term in the random features, defaults to False.
		device : str
			Which device to use, can be 'cpu' or 'cuda', defaults to None which means use cuda if available.
		nu : float
			Smoothness parameter for the Matern kernel, must be greater than 0. Defaults to None.
		"""
		assert nu is not None
		assert nu>0
		self.nu = nu
		super().__init__(*args, **kwargs)

	def set_W2(self, W2=None):
		if W2 is not None:
			self.W2=W2
		else:
			df=2*self.nu
			chi2_dist = Chi2(df=df)
			chi2_samples = chi2_dist.sample((self._num_features,))/(2*self.nu)
			self.W2 = torch.sqrt(chi2_samples).to(self.device)

	def apply_W2(self, XW1):
		if self.bias_term:
			return self.c1 * ((XW1/self.W2) + self._bias).cos()
		else:
			return self.c1 * torch.cat([(XW1/self.W2).cos(), (XW1/self.W2).sin()], dim=-1)