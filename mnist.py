from tinygrad import Tensor, dtypes
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
import numpy as np 
from typing import List
from pathlib import Path
import os

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

# TODO: write data loader from scratch, tinygrad one is really weird, don't actually know what it's doing
def load_mnist() -> List[Tensor]:
  tensors = []
  for file in sorted(os.listdir("data")):
    with open(Path.cwd() / "data" / file, "rb") as f:
      if "label" not in f.name:
        tensors.append(Tensor(np.frombuffer(f.read(), dtype=np.uint8)[0x10:].reshape((-1, 28*28)).astype(np.float32), requires_grad=False).reshape(-1, 1, 28, 28))
      else:
        tensors.append(Tensor(np.frombuffer(f.read(), dtype=np.uint8)[8:].astype(np.float32), requires_grad=False))
  return tensors

# import matplotlib.pyplot as plt
# plt.imshow(data, cmap='gray')
# plt.show()

# full model
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

net = TinyNet()
opt = SGD([net.l1.weight, net.l2.weight], lr=3e-4)

X_train= load_mnist()[0]
Y_train = load_mnist()[1]
X_test = load_mnist()[2]
Y_test = load_mnist()[3]
print(X_train.shape)

with Tensor.train():
  for step in range(1000):
    # random sample a batch
    samp = np.random.randint(0,10000)
    batch = X_train[samp]
    # get the corresponding labels
    labels = Y_train[samp]

    # forward pass
    out = net(batch)

    # compute loss
    loss = sparse_categorical_crossentropy(out, labels)

    # zero gradients
    opt.zero_grad()

    # backward pass
    loss.backward()

    # update parameters
    opt.step()

    # calculate accuracy
    pred = out.argmax(axis=-1)
    acc = (pred == labels).mean()

    if step % 100 == 0:
      print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy()}")
