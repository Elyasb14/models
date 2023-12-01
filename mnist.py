from tinygrad.tensor import Tensor
from tinygrad.helpers import dtypes
from tinygrad.nn import Linear
import numpy as np 
import os
from tinygrad.nn.optim import SGD
import matplotlib.pyplot as plt

def plot_mnist(idx):
  data = TRAIN_IM[2].reshape(28,28)
  print(TRAIN_LAB[2])
  plt.imshow(data)
  plt.savefig("plot")

# TODO: implement this myself
def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

def load_mnist() -> tuple(np.ndarray):
  # anon function that takes in bytes and loads them into 1d numpy array
  parse = lambda file: np.frombuffer(file, dtype=np.uint8).copy()
  data_dir = sorted(os.listdir("data"))
  with open(f"data/{data_dir[0]}", "rb") as f: TEST_IM = parse(f.read())[0x10:].reshape((-1, 28*28)).astype(np.float32)
  with open(f"data/{data_dir[1]}", "rb") as f: TEST_LAB = parse(f.read())[8:]
  with open(f"data/{data_dir[2]}", "rb") as f: TRAIN_IM = parse(f.read())[0x10:].reshape((-1, 28*28)).astype(np.float32)
  with open(f"data/{data_dir[3]}", "rb") as f: TRAIN_LAB = parse(f.read())[8:]
  return TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB

class TinyNet:
  def __init__(self):
    # l1 outfeatures == l2 in features
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)
  
  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x
  
net = TinyNet()  
opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)
TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist()

# with Tensor.train():
#   for step in range(1000):
#     # gives array of 64 random integers
#     samp = np.random.randint(0, TRAIN_IM.shape[0], size=(64))
#     batch = Tensor(TRAIN_IM[samp], requires_grad=False)
#     labels = Tensor(TRAIN_LAB[samp])
#     # i need to understand everything from this comment down, i think i get everything above this comment
#     out = net(batch)
#     loss = sparse_categorical_crossentropy(out, labels)
#     opt.zero_grad()
#     loss.backward()
#     opt.step()
#     pred = out.argmax(axis=-1)
#     acc = (pred == labels).mean()
#     if step % 100 == 0:
#       print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")
