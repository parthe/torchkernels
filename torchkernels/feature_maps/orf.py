import numpy as np
import scipy.stats as stats
import torch

class ORF:
	def __init__(self, 
		input_dim:int, 
		num_features:int, 
		length_scale:float=1., 
		bias_term:bool=False, 
		device:str=None):
		"""Initialize an instance of the ORF class.
		
		Parameters
		----------
		input_dim : int
		  input dimension of the data.
		num_features : int
		  number of random features to generate.
		length_scale : float
		  kernel length scale, defaults to 1.
		bias_term : bool
		  whether to include a bias term in the random features, defaults to False.
		device : str
		  which device to use, can be 'cpu' or 'cuda', defaults to None which means use cuda if available.
		"""
		self.input_dim = input_dim
		self.num_features = num_features
		self.length_scale = length_scale
		self.bias_term = bias_term
		if device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device(device)
		self.c1 = (torch.sqrt(torch.tensor(2 / self.num_features)).to(self.device))
		if not self.bias_term:
			self.num_features = self.num_features//2

		Q_arr = []
		for _ in range(int(np.ceil(self.num_features/self.input_dim))):
			Q = stats.ortho_group.rvs(dim=self.input_dim)
			Q_arr.append(Q)
		self.Q = np.concatenate(Q_arr, axis=0)[:self.num_features].T
		del Q_arr
		self.Q = torch.from_numpy(self.Q).to(torch.float32).to(self.device)

		self.set_S()
		if self.bias_term:
			self._bias = ((torch.rand(self.num_features) * torch.pi * 2).to(self.device))

	def __call__(self, x):
		x = x.to(self.device)
		if self.bias_term:
			return self.c1 * ((torch.mm(x, self.Q)*self.S) + self._bias).cos()
		else:
			return self.c1 * torch.cat([(torch.mm(x, self.Q)*self.S).cos(), (torch.mm(x, self.Q)*self.S).sin()], dim=-1)

	def set_Q(self, Q=None):
		if Q is not None:
			self.Q = Q

	def set_S(self):
		raise NotImplementedError("This method must be implemented in the subclass")