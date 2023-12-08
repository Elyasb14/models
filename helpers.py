import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Tuple
from tinygrad import Tensor

def plot_mnist(idx):
    TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB = load_mnist()
    data = TRAIN_IM[idx].reshape(28,28)
    print(TRAIN_LAB[idx])
    plt.imshow(data)
    plt.savefig("plot")

def load_fashion(tensors=False) -> Tuple[np.ndarray]:
  def parse(file): return np.frombuffer(file, dtype=np.uint8).copy() # gives 1d array of 64 random integers
  data_dir = sorted(os.listdir("data/fashionmnist"))
  with open(f"data/fashionmnist/{data_dir[0]}", "rb") as f: TEST_IM = parse(f.read())[0x10:].reshape((-1, 28*28)).astype(np.float32)
  with open(f"data/fashionmnist/{data_dir[1]}", "rb") as f: TEST_LAB = parse(f.read())[8:]
  with open(f"data/fashionmnist/{data_dir[2]}", "rb") as f: TRAIN_IM = parse(f.read())[0x10:].reshape((-1, 28*28)).astype(np.float32)
  with open(f"data/fashionmnist/{data_dir[3]}", "rb") as f: TRAIN_LAB = parse(f.read())[8:]
  if tensors: return Tensor(TRAIN_IM).reshape(-1, 1, 28, 28), Tensor(TRAIN_LAB), Tensor(TEST_IM).reshape(-1, 1, 28, 28), Tensor(TEST_LAB)   # noqa: E701
  return TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB

def load_mnist(tensors=False) -> Tuple[np.ndarray]:
  def parse(file): return np.frombuffer(file, dtype=np.uint8).copy() # gives 1d array of 64 random integers
  data_dir = sorted(os.listdir("data/mnist"))
  with open(f"data/mnist/{data_dir[0]}", "rb") as f: TEST_IM = parse(f.read())[0x10:].reshape((-1, 28*28)).astype(np.float32)
  with open(f"data/mnist/{data_dir[1]}", "rb") as f: TEST_LAB = parse(f.read())[8:]
  with open(f"data/mnist/{data_dir[2]}", "rb") as f: TRAIN_IM = parse(f.read())[0x10:].reshape((-1, 28*28)).astype(np.float32)
  with open(f"data/mnist/{data_dir[3]}", "rb") as f: TRAIN_LAB = parse(f.read())[8:]
  if tensors: return Tensor(TRAIN_IM).reshape(-1, 1, 28, 28), Tensor(TRAIN_LAB), Tensor(TEST_IM).reshape(-1, 1, 28, 28), Tensor(TEST_LAB)   # noqa: E701
  return TEST_IM, TEST_LAB, TRAIN_IM, TRAIN_LAB