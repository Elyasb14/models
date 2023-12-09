from helpers import load_mnist
from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from tinygrad.nn.optim import SGD
from tinygrad.nn import Linear
from tinygrad.nn.state import get_parameters
import numpy as np
from tinygrad.jit import TinyJit

TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist()

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)

