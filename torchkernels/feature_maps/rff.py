class RFF:

  def __init__():
    self.W1 = ...
    self.set_W2()
    
  def __call__():
    return self.apply_W2(x @ self.W1)

  def set_W2(self):
    raise NotImplementedError("This method must be implemented in the subclass")

  def apply_W2(self):
    raise NotImplementedError("This method must be implemented in the subclass")