from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import numpy as np 

# TODO: write data loader from scratch, tinygrad one is really weird, don't actually know what it's doing

class TinyNet:
  def __init__(self):
    # l1 outfeatures == ls in features
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)
  
  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x