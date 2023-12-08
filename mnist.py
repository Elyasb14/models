from tinygrad.helpers import dtypes, Timing
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import numpy as np
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, safe_save, safe_load, get_state_dict, load_state_dict
from helpers import load_fashion, load_mnist
import matplotlib.pyplot as plt
from random import randint

def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
    loss_mask = Y != ignore_index
    y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
    y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
    return self.log_softmax().mul(y).sum() / loss_mask.sum()

TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist()

class TinyNet:
  def __init__(self):
    self.l1 = Linear(784, 128, bias=False)
    self.l2 = Linear(128, 10, bias=False)

  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x)
    x = x.leakyrelu()
    x = self.l2(x)
    return x

def train(steps=10):
  net = TinyNet() 
  opt = SGD(get_parameters(net), lr=3e-4) # can also do SGD([net.l1.weight, net.l2.weight], lr=3e-4)
  loss_list =[]
  with Tensor.train():
    for step in range(steps):
      samp = np.random.randint(0, TRAIN_IM.shape[0], size=(64))
      batch = Tensor(TRAIN_IM[samp], requires_grad=False) # grad is false because we don't need to compute gradients on these tensors
      labels = Tensor(TRAIN_LAB[samp], requires_grad=False)
      out = net(batch) # forward pass
      loss = sparse_categorical_crossentropy(out, labels) # TODO: it would be nice to implement this myself, i don't really know what is going on here tbh 
      opt.zero_grad()
      loss.backward() # can only be called on scalar tensors, loss must give a scalar tensor
      opt.step() # this updates the parameters
      pred = out.argmax(axis=-1) # this basically return the model's prediction 
      acc = (pred == labels).mean() # i'm having a really hard time understanding what's going on here
      loss_list.append(loss)
      if step % 100 == 0:
        print(f"Step {step+1} | Loss: {loss.numpy()} | Accuracy: {acc.numpy() * 100}") # prints step, loss, and accuracy
  with Timing("Time: "):
    avg_acc = 0
    for step in range(steps):
      samp = np.random.randint(0, TEST_IM.shape[0], size=(64))
      batch = Tensor(TEST_IM[samp], requires_grad=False)
      labels = TEST_LAB[samp]
      out = net(batch)
      pred = out.argmax(axis=-1).numpy()
      avg_acc += (pred == labels).mean()
    print(f"Test Accuracy: {avg_acc/steps}")
  state_dict = get_state_dict(net)
  safe_save(state_dict, "model.safetensors")
  plt.plot([loss.numpy() for loss in loss_list])
  plt.xlabel("steps")
  plt.ylabel("loss")
  plt.title("loss/step")
  plt.savefig("loss")


def inference():
  index = randint(0, TEST_IM.shape[0])
  model = TinyNet()
  state_dict = safe_load("model.safetensors")
  load_state_dict(model, state_dict) # this updates the models internal state_dict, holding information about the current weights
  example = Tensor(TEST_IM[index], requires_grad=False)
  example_label = TEST_LAB[index]
  prediction = model(example).argmax(axis=-1).numpy()
  print(f"actual label: {example_label}, guess: {prediction}")
  plt.imshow(example.numpy().reshape((28,28)))
  plt.savefig("plot")

if __name__ == "__main__":
  train(steps=1000)
  inference()