import torch
import numpy as np



class KernelLinearOperator:
  def __init__(self, K, X, Z=None, dtype=None, x_batch=None, z_batch=None):
    # super(self,torch.Tensor).__init__()
    self.K = K
    self.X = X
    self.Z = X if Z is None else Z
    self.dtype = dtype or X.dtype
    self.device = X.device
    self.x_batch = x_batch
    self.z_batch = z_batch
    self.shape = (len(self.X), len(self.Z))

  def _apply(self, X, Z, V, x_batch, z_batch):
    V = torch.as_tensor(V, device=self.device, dtype=self.dtype)

    if V.ndim == 1:
      V2 = V[:, None]
      squeeze = True
    else:
      V2 = V
      squeeze = False

    m, n = len(X), len(Z)
    assert V2.shape[0] == n

    out = torch.zeros((m, V2.shape[1]), device=self.device, dtype=self.dtype)

    xb = x_batch or m
    zb = z_batch or n

    for i in range(0, m, xb):
      Xi = X[i:i + xb]
      out_i = out[i:i + xb]

      for j in range(0, n, zb):
        Zj = Z[j:j + zb]
        kij = self.K(Xi, Zj).to(device=self.device, dtype=self.dtype)
        out_i += kij @ V2[j:j + zb]

    return out[:, 0] if squeeze else out

  def matvec(self, v):
    return self @ v

  def matmat(self, V):
    return self @ V

  def __matmul__(self, V):
    return self._apply(self.X, self.Z, V, self.x_batch, self.z_batch)

  def __rmatmul__(self, U):
    U = torch.as_tensor(U, device=self.device, dtype=self.dtype)

    if U.ndim == 1:
      return self.T @ U

    return (self.T @ U.T).T

  def _resolve(self, idx, base):
    return base[idx]

  def __getitem__(self, key):
    if not isinstance(key, tuple):
      row_key = key
      col_key = slice(None)
    else:
      row_key, col_key = key

    rows = self._resolve(row_key, torch.arange(len(self.X)))
    cols = self._resolve(col_key, torch.arange(len(self.Z)))

    rows = np.atleast_1d(rows)
    cols = np.atleast_1d(cols)

    return KernelLinearOperator(
      self.K,
      self.X[rows],
      self.Z[cols],
      dtype=self.dtype,
      x_batch=self.x_batch,
      z_batch=self.z_batch,
    )

  @property
  def T(self):
    return KernelLinearOperator(
      self.K,
      self.Z,
      self.X,
      dtype=self.dtype,
      x_batch=self.z_batch,
      z_batch=self.x_batch,
    )

  def diag(self):
    return torch.tensor([self.K(x.unsqueeze(0),z.unsqueeze(0)) for x,z in zip(self.X,self.Z)])

  @property
  def H(self):
    return self.T

  def __array__(self, *args, **kwargs):
    raise TypeError("KernelLinearOperator is matrix-free and cannot be converted to a NumPy array.")

