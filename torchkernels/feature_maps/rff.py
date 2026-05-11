import torch
import math
import numpy as np

class RFF:

	def __init__(self, 
		input_dim:int, 
		num_features:int, 
		length_scale:float=1., 
		shape_matrix:torch.Tensor=None,
		shape_matrix_sqrtm_fn=torch.linalg.cholesky, # cholesky returns L of shape (d,d) where M = L @ L.T, we want x.T @ M @ z which we apply a as (x @ L) for x of shape (batch_size,d)
		bias_term:bool=False, 
		device:str=None,
		dtype=torch.float32,
		seed:int = None):	
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
			shape matrix for the random features, defaults to None. Shape matrix entered must be symmetric, positive definite and of dimension d x d where d is input dimension.
		bias_term : bool
			whether to include a bias term in the random features, defaults to False.
		device : str
			which device to use, can be 'cpu' or 'cuda', defaults to None which means use cuda if available.
		dtype : torch.dtype
			data type to use, defaults to torch.float32.
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
		self.dtype = dtype
		self.seed = seed
		self.shape_matrix_sqrtm_fn = shape_matrix_sqrtm_fn
	
		self.set_shape_matrix_sqrtm(shape_matrix)
		self.bias_term = bias_term

		self.c1 = torch.sqrt(torch.tensor(2 / self.num_features)).to(self.dtype).to(self.device)
		if not self.bias_term:
			self._num_features = self.num_features//2
		else: 
			self._num_features = self.num_features
		if self.seed is not None:
			self.torch_gen = torch.Generator(device=self.device).manual_seed(self.seed)
		else: 
			self.torch_gen = torch.Generator(device=self.device)
		self.W1 = torch.randn(self.input_dim,self._num_features,device=self.device, dtype=self.dtype, generator=self.torch_gen)
		self.set_W2()
		if self.bias_term:
			self._bias = (torch.rand(
				self._num_features, dtype=self.dtype, generator=self.torch_gen
			) * math.pi * 2).to(self.dtype).to(self.device)
	
	
	def set_W1(self, W1=None):
		if W1 is not None:
			self.W1=W1

	def set_shape_matrix_sqrtm(self,shape_matrix):
		if shape_matrix is not None:
			self.shape_matrix = shape_matrix.to(self.device).to(self.dtype)
			assert self.shape_matrix.shape[0] == self.shape_matrix.shape[1] == self.input_dim, "Shape matrix must be square and of dimension d x d where d is input dimension."
			self.shape_matrix_sqrtm = self.shape_matrix_sqrtm_fn(self.shape_matrix)
		else:
			self.shape_matrix = None
			self.shape_matrix_sqrtm = None

	def set_length_scale(self,ell):
		self.length_scale = ell

	def set_W2(self):
		raise NotImplementedError("This method must be implemented in the subclass")

	def apply_W2(self):
		raise NotImplementedError("This method must be implemented in the subclass")

	def apply_W1(self, x):
		if self.shape_matrix is not None:
			x = torch.mm(x, self.shape_matrix_sqrtm)
		return torch.mm(x/self.length_scale, self.W1)

	def __call__(self, x):
		assert x.dtype == self.dtype, "Input must be of type {}, else specify dtype parameter".format(self.dtype) 
		assert x.shape[1] == self.input_dim, "Input must have dimension {}".format(self.input_dim)
		x = x.to(self.device)
		return self.apply_W2(self.apply_W1(x))
