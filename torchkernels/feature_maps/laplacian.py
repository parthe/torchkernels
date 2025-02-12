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
  phi = LaplacianORF()
  Phi1 = phi(X1)
  Phi2 = phi(X2)