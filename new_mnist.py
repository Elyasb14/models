from helpers import load_mnist
from tinygrad.tensor import Tensor
import tinygrad.nn as nn
import numpy as np


TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist(tensors=False)

class TinyNet:
  def __init__(self):
    self.l1 = nn.Linear(784, 128, bias=False)
    self.l2 = nn.Linear(128, 10, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x

def train(steps: int, samp=np.random.randint(0, TRAIN_IM.shape[0], size=(64))):
  model = TinyNet()
  opt = nn.optim.SGD(nn.state.get_parameters(model), lr=3e-4)




def inference():
  pass


if __name__ == "__main__":
  train(1000)

