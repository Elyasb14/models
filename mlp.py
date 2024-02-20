from helpers import load_fashion, load_mnist, plot_loss
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.optim import SGD
from tinygrad.nn.state import get_parameters, get_state_dict, safe_save, safe_load, load_state_dict
from random import randint
from tinygrad.jit import TinyJit
from tqdm import trange
import argparse

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
  loss_list = []
  with Tensor.train():
    for i in (t:=trange(steps)):
      samp = [randint(0, TRAIN_IM.shape[0]-1) for _ in range(64)]
      batch, batch_labels = Tensor(TRAIN_IM[samp], requires_grad=False), Tensor(TRAIN_LAB[samp], requires_grad=False)
      opt.zero_grad()
      loss = model(batch).sparse_categorical_crossentropy(batch_labels).backward()
      opt.step()
      loss_list.append(loss)
      if i % 100 == 0:
        t.set_description(f"loss: {loss.item()}")
  plot_loss(loss_list)
  safe_save(get_state_dict(model), "models/mlp.safetensors")

@TinyJit
def evaluate(steps):
  avg_acc = 0
  for i in (t:=trange(steps)):
    samp = [randint(0, TEST_IM.shape[0]-1) for _ in range(64)]
    test_pred = model(Tensor(TEST_IM[samp], requires_grad=False)).argmax(axis=-1)
    acc = (test_pred == Tensor(TEST_LAB[samp], requires_grad=False)).mean()*100
    avg_acc += acc.item()
    t.set_description(f"avg acc: {avg_acc/steps}%")

@TinyJit
def inference():
  samp = randint(0, 1000)
  weights = safe_load("models/mlp.safetensors")
  load_state_dict(model, weights)
  pred, label = model(Tensor(TEST_IM[samp])).argmax(axis=-1).item(), TEST_LAB[samp]
  print(f"model's prediction: {pred}, actual label: {label}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="arguments for training/infering on fashion or regular mnist", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dataset", type=str, required=True, help="choose between mnist and fashion mnist")
  parser.add_argument("--train", action="store_true", help="to train or not to train")
  parser.add_argument("--infer", action="store_true", help="infer on a random image from the test dataset")
  args = parser.parse_args()
  if args.dataset == "mnist":
    TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist(tensors=False) 
  elif args.dataset == "fashion":
    TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_fashion(tensors=False)
  if args.train:
    train(1000)
    evaluate(1000)
  elif args.infer:
    inference()
