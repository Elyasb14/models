from helpers import plot_loss, load_mnist, load_fashion
from tinygrad.nn import Conv2d, BatchNorm2d, Linear
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters, safe_save, get_state_dict, safe_load, load_state_dict
from tinygrad.tensor import Tensor
from tinygrad.helpers import Timing
from tqdm import trange
import argparse

class Cnn:
  def __init__(self):
    self.layers = [
      Conv2d(1, 32, 5), Tensor.relu,
      Conv2d(32, 32, 5), Tensor.relu,
      BatchNorm2d(32), Tensor.max_pool2d,
      Conv2d(32, 64, 3), Tensor.relu,
      Conv2d(64, 64, 3), Tensor.relu,
      BatchNorm2d(64), Tensor.max_pool2d,
      lambda x: x.flatten(1), Linear(576, 10)
    ]

  def __call__(self, x: Tensor) -> Tensor: return x.sequential(self.layers)

model = Cnn()
opt = Adam(get_parameters(model))

def train(steps):
  losses = []
  with Timing("time to train: "):
    with Tensor.train():
      for i in (t:=trange(steps)):
        samp = Tensor.randint(512, high=TRAIN_IM.shape[0])
        batch, labels = TRAIN_IM[samp], TRAIN_LAB[samp]
        opt.zero_grad()
        loss = model(batch).sparse_categorical_crossentropy(labels).backward()
        opt.step()
        t.set_description(f"loss: {loss.item():6.2f}")
        losses.append(loss)
  safe_save(get_state_dict(model), "models/cnn.safetensors")
  plot_loss(losses)

def evaluate(steps):
  avg_acc = 0
  with Timing("time to evaluate: "):
    for i in (t:=trange(steps)):
      samp = Tensor.randint(512, high=TEST_IM.shape[0])
      test_pred, labels  = model(TEST_IM[samp]).argmax(axis=1), TEST_LAB[samp]
      acc = (test_pred == labels).mean()*100
      avg_acc += acc.item()
      t.set_description(f"avg acc: {avg_acc/steps:5.2f}%")

def inference():
  model, weights = Cnn(), safe_load("models/cnn.safetensors")
  sample = Tensor.randint(1, high=TEST_IM.shape[0])
  load_state_dict(model, weights)
  out, label = model(TEST_IM[sample]).argmax(axis=1), TEST_LAB[sample]
  print(f"model pred: {out.realize().item()}, actual label: {label.realize().item()}")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="arguments for training/infering on fashion or regular mnist", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dataset", type=str, required=True, help="choose between mnist and fashion mnist")
  parser.add_argument("--train", action="store_true", help="to train or not to train")
  parser.add_argument("--infer", action="store_true", help="infer on a random image from the test dataset")
  args = parser.parse_args()
  if args.dataset == "mnist":
    TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist(tensors=True) 
  elif args.dataset == "fashion":
    TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_fashion(tensors=True)
  if args.train:
    train(70)
    evaluate(70)
  elif args.infer:
    inference()
