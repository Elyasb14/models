from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from tinygrad.nn import Linear
import numpy as np
import os
from tinygrad.nn.optim import SGD
import matplotlib.pyplot as plt
from typing import Tuple

def plot_mnist(idx):
  data = TRAIN_IM[idx].reshape(28,28)
  print(TRAIN_LAB[idx])
  plt.imshow(data)
  plt.savefig("plot")

# TODO: implement this myself
# def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
#     loss_mask = Y != ignore_index
#     y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
#     y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
#     return self.log_softmax().mul(y).sum() / loss_mask.sum()

def load_mnist() -> Tuple[np.ndarray]:
  parse = lambda file: np.frombuffer(file, dtype=np.uint8).copy() # gives 1d array of 64 random integers
  data_dir = sorted(os.listdir("data"))
  with open(f"data/{data_dir[0]}", "rb") as f: TEST_IM = parse(f.read())[0x10:].reshape((-1, 28*28)).astype(np.float32)
  with open(f"data/{data_dir[1]}", "rb") as f: TEST_LAB = parse(f.read())[8:]
  with open(f"data/{data_dir[2]}", "rb") as f: TRAIN_IM = parse(f.read())[0x10:].reshape((-1, 28*28)).astype(np.float32)
  with open(f"data/{data_dir[3]}", "rb") as f: TRAIN_LAB = parse(f.read())[8:]
  return TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB

TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist()

class TinyNet:
  def __init__(self):
    # l1 outfeatures == l2 in features
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)
  
  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x)
    x = x.cosh()
    x = self.l2(x)
    return x
  
def train():
  net = TinyNet()  
  opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)
  with Tensor.train():
    for step in range(1):
      samp = np.random.randint(0, TRAIN_IM.shape[0], size=(64)) # gives 1d array of 64 random integers
      batch = Tensor(TRAIN_IM[samp], requires_grad=False)
      labels = Tensor(TRAIN_LAB[samp])
      print(batch[0])
      print(labels)


if __name__ == "__main__":
  train()