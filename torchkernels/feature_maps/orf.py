import numpy as np
import scipy.stats as stats
import torch
import math

class ORF:
	def __init__(self, 
		input_dim:int, 
		num_features:int, 
		length_scale:float=1.,
		shape_matrix:torch.Tensor=None,
		bias_term:bool=False, 
		device:str=None, 
		float_type=torch.float32,
		seed:int = None):
		"""Initialize an instance of the ORF class.
		
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
		  float type to use, defaults to torch.float32.
		seed : int
		  seed, type int. Defaults to None.
		"""
		if device is None:
			self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		else:
			self.device = torch.device(device)

		self.input_dim = input_dim
		self.num_features = num_features
		self.length_scale = length_scale
		self.float_type = float_type
		self.seed = seed
		
		if shape_matrix is not None:
			self.shape_matrix = shape_matrix.to(self.float_type)
			assert self.shape_matrix.shape[0] == self.shape_matrix.shape[1] == self.input_dim, "Shape matrix must be square and of dimension d x d where d is input dimension."
			self.sqrt_M = torch.linalg.cholesky(self.shape_matrix.to(self.device))
		else:
			self.sqrt_M = None
			self.shape_matrix = None
		self.bias_term = bias_term

		self.c1 = (torch.sqrt(torch.tensor(2 / self.num_features)).to(self.float_type).to(self.device))
		if not self.bias_term:
			self._num_features = self.num_features//2
		else: 
			self._num_features = self.num_features

		Q_arr = []
		Q_rv = stats.ortho_group(dim=self.input_dim, seed=self.seed)
		for _ in range(int(np.ceil(self._num_features/self.input_dim))):
			Q = Q_rv.rvs()
			Q_arr.append(Q)
		self.Q = np.concatenate(Q_arr, axis=0)[:self._num_features].T
		del Q_arr
		self.Q = torch.from_numpy(self.Q).to(self.float_type).to(self.device)
		if self.seed is not None:
			self.torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)
		else: self.torch_gen = torch.Generator(device=self.device)

		self.set_S()
		if self.bias_term:
			self._bias = ((torch.rand(self._num_features, dtype=self.float_type) * math.pi * 2).to(self.device))

	def __call__(self, x):
		assert x.dtype == self.float_type, "Input must be of type {}, else specify float_type parameter".format(self.float_type) 
		assert x.shape[1] == self.input_dim, "Input must have dimension {}".format(self.input_dim)
		x = x.to(self.device)
		if self.shape_matrix is not None:
			x = torch.mm(x, self.sqrt_M)
		if self.bias_term:
			return self.c1 * ((torch.mm(x, self.Q)*self.S) + self._bias).cos()
		else:
			return self.c1 * torch.cat([(torch.mm(x, self.Q)*self.S).cos(), (torch.mm(x, self.Q)*self.S).sin()], dim=-1)

	def set_Q(self, Q=None):
		if Q is not None:
			self.Q = Q

	def set_S(self):
		raise NotImplementedError("This method must be implemented in the subclass")