from .orf import ORF
from .rff import RFF
import scipy.stats as stats
import numpy as np
import torch


class GaussianORF(ORF):
	def set_S(self, S=None):
		if S is not None:
			self.S=S
		else:
			self.S = torch.from_numpy(stats.chi.rvs(self.input_dim,size=self._num_features)
                            /self.length_scale).to(self.float_type).to(self.device)


class GaussianRFF(RFF):
	def set_W2(self, W2=None):
		if W2 is not None:
			self.W2=W2
		else:self.W2 = None

	def apply_W2(self, XW1):
		if self.bias_term:
			return self.c1 * (XW1 + self._bias).cos()
		else:
			return self.c1 * torch.cat([XW1.cos(), XW1.sin()], dim=-1)