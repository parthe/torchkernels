from .linalg.fmm import KmV
from torch import nn

class KernelModel(nn.Module):
    
    def __init__(self, kernel_fn, centers, weights=None, tasks=None):
        self.kernel = kernel_fn
        self.centers = centers
        self.size, self.dim = (len(centers), 1) if len(centers.shape)==1 else centers.shape
        self.weights = weights
        self.tasks = tasks
        if weights is not None:
            p_, t_ = (len(weights), 1) if len(weights.shape)==1 else weights.shape
            assert p_==self.size, "number of centers and number of weights do not match"
            if tasks is None: self.tasks = t_
    
    def forward(self, samples):
        return KmV(self.kernel, samples, self.centers, self.weights)

    def matrix(self, samples):
        return self.kernel(samples, self.centers)
    
    def fit(self, samples, labels, reg=0., method='solve'):
        n, c = (len(y), 1) if len(labels.shape)==1 else labels.shape
        n_, d_ = len(samples), 1 if len(samples.shape)==1 else samples.shape
        self.tasks = c_ if self.tasks is None else self.tasks
        assert n_==n, "number of samples do not match number of labels"
        assert d_==self.dim, "number of dimensions of samples and model centers do not match"
        assert self.tasks==c_, "number of tasks of labels and model weights do not match"
        if (samples is self.centers) or (samples is None):
            self.fit_solve(labels, reg)
        else:
            self.fit_lstsq(samples, labels, reg=0.)
      
    def fit_lstsq(self, samples, labels, reg=0.):
        Kmat = self.kernel(samples, centers)
        self.weights = torch.linalg.lstsq(Kmat, y)

    def fit_solve(self, labels, reg=0.):
        Kmat = self.kernel(samples, centers)
        raise NotImplementedError("Incomplete")
        Kty = self.
        self.weights = torch.linalg.solve(Kmat + reg*torch.eye(self.size), labels)
  
