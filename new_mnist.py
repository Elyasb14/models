from helpers import load_mnist, plot_loss
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters
from random import randint
from tinygrad.jit import TinyJit

TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist(tensors=True)

class TinyNet:
  def __init__(self):
    self.layers = [
      Linear(784, 128, bias=False),
      Tensor.leakyrelu,
      Linear(128, 10, bias=False)
      ]

  def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)

model = TinyNet()
opt = SGD(get_parameters(model), lr=3e-4)

@TinyJit
def train(steps):
  losses = []
  with Tensor.train():
    for i in range(steps):
      samp = Tensor([randint(0, TEST_IM.shape[0]) for _ in range(64)])
      batch, batch_labels = TRAIN_IM[samp], TRAIN_LAB[samp]
      opt.zero_grad()
      loss = model(batch).sparse_categorical_crossentropy(batch_labels).backward()
      losses.append(loss)
      opt.step()
      if i % 100 == 0:
        print(f"test accuracy %: , loss: {loss.numpy()}")
  plot_loss(losses)

@TinyJit
def evaluate(steps):
  avg_acc = 0
  for i in range(steps):
    samp = Tensor([randint(0, TEST_IM.shape[0]) for _ in range(64)])
    test_pred = model(TEST_IM[samp]).argmax(axis=-1)
    acc = (test_pred == TEST_LAB[samp]).mean()*100
    avg_acc += acc.numpy()
  return (avg_acc/steps)

if __name__ == "__main__":
  train(1000)
  print(evaluate(1000))