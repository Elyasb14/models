import requests
import gzip


def fetch_mnist(url: str) -> None:

    "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    X_TRAINf"{BASE_URL}train-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_train = parse(fetch(f"{BASE_URL}train-labels-idx1-ubyte.gz"))[8:]
    X_test = parse(fetch(f"{BASE_URL}t10k-images-idx3-ubyte.gz"))[0x10:].reshape((-1, 28*28)).astype(np.float32)
    Y_test = parse(fetch(f"{BASE_URL}t10k-labels-idx1-ubyte.gz"))[8:]


