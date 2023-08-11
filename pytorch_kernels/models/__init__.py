from .linalg.fmm import KmV


class KernelModel(nn.Module):
    
    def __init__(self, kernel_fn, centers, weights=None, tasks=None):
        self.kernel, self.centers = kernel_fn, centers
        self.size, self.dim = (len(centers), 1) if len(centers.shape)==1 else centers.shape
        self.weights, self.tasks = weights, tasks
        if weights is not None:
            p_, t_ = (len(weights), 1) if len(weights.shape)==1 else weights.shape
            assert p_ == self.size, "number of centers and number of weights do not match"
            self.tasks = t_
            if tasks is not None:
                assert tasks == self.tasks, "number of tasks in weights provided do not match `tasks`"

    def __call__(self, samples):
        return self.predict(samples)
    
    def predict(self, samples):
        return KmV(self.kernel, samples, self.centers, self.weights)
    
    def fit(self, labels, reg=0.):
        """
        solves the kernel regression problem (K + reg*I) a = labels
        """
        n, c = (len(y), 1) if len(labels.shape)==1 else labels.shape
        self.tasks = c if self.tasks is None else self.tasks
        assert n==self.size, "number of samples in (labels) and (self.centers) do not match"
        assert c==self.tasks, "number of tasks in (labels) and (self.tasks) do not match"
        kmat = self.kernel(self.centers)
        self.weights = torch.linalg.solve(kmat + reg*torch.eye(n, dtype=kmat.dtype), labels.type(kmat.type()))

    def score(self, samples, labels, score_fn):
        return score_fn(self.forward(samples), labels)
  
