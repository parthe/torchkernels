class ORF:

  def __init__(input_dim,num_features):
    self.Q = ...
    self.set_S()

  def __call__(x):
    return (x @ self.Q) * self.S

  def set_Q(Q):
    self.Q = Q

  def set_S():
    raise NotImplementedError(
      "This method must be implemented in the subclass")