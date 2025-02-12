from .orf import ORF
from .rff import RFF

class LaplacianORF(ORF):

  def set_S():
    self.S = ...

class LaplacianRFF(RFF):
  
  def set_W2():
    self.W2 = ...

  def apply_W2(self):
    return ...

if __name__=="__main__":
  feature_map = LaplacianORF()
  n,p,d=10,2,7
  X = torch.randn(n,d)
  W1 = torch.randn(d,p)
  W2 = torch.randn(p)
  feature_map.set_W1(W1)
  feature_map.set_W2(W2)
  assert torch.allclose(feature_map(X),sincos((X@W1)/W2))

