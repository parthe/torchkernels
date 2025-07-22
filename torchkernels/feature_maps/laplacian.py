import random
from .orf import ORF
from .rff import RFF
import scipy.stats as stats
import numpy as np
from numpy.random import default_rng
import torch


class LaplacianORF(ORF):

	def set_S(self, S=None):
		if S is not None:
			self.S=S
		else:
			self.S = torch.from_numpy(np.sqrt(stats.betaprime.rvs(self.input_dim/2,1/2, size=self._num_features, random_state=self.seed))/
                             self.length_scale).to(self.dtype).to(self.device)
   

class LaplacianORF_QMC(ORF):

	def set_S(self, S=None):
		if S is not None:
			self.S=S
		else:
			sampler = stats.qmc.Halton(d=1, scramble=True, seed=self.seed)
			u = sampler.random(self._num_features).flatten()
			samples = stats.betaprime.ppf(u, self.input_dim/2, 1/2)
			self.S = torch.from_numpy(np.sqrt(samples)/self.length_scale).to(self.dtype).to(self.device)

class LaplacianRFF(RFF):

	def set_W2(self, W2=None):
		if W2 is not None:
			self.W2=W2
		else:
			
			self.W2 = torch.randn(self._num_features, dtype=self.dtype, generator=self.torch_gen, device=self.device)

	def apply_W2(self, XW1):
		if self.bias_term:
			return self.c1 * ((XW1/self.W2) + self._bias).cos()
		else:
			return self.c1 * torch.cat([(XW1/self.W2).cos(), (XW1/self.W2).sin()], dim=-1)