from .orf import ORF
from .rff import RFF
import scipy.stats as stats
import numpy as np
import torch


class LaplacianORF(ORF):

	def set_S(self):
		self.S = torch.from_numpy(np.sqrt(stats.betaprime.rvs(self.input_dim/2,1/2, size=self.num_features))/self.length_scale).to(self.device)
	
	def test_set_S(self, S):
		self.S = S	

class LaplacianRFF(RFF):

	def set_W2(self):
		self.W2 = torch.randn(self.num_features).to(self.device)

	def test_set_W2(self, W2):
		self.W2 = W2
	
	def apply_W2(self, XW1):
		if self.bias_term:
			return self.c1 * ((XW1/self.W2) + self._bias).cos()
		else:
			return self.c1 * torch.cat([(XW1/self.W2).cos(), (XW1/self.W2).sin()], dim=-1)