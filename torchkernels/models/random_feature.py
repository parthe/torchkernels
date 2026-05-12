import torch
from ..feature_maps.rff import RFF

class RandomFeatureModel(RFF):
    def __init__(self, out_dim=1, bias=True, **kwargs):
      super().__init__(**kwargs)
      self.weights = torch.zeros(self.num_features, out_dim)
      self.bias = torch.zeros(out_dim)

    def __call__(self, samples):
        return self.feature_map(samples) @ self.weights + self.bias
