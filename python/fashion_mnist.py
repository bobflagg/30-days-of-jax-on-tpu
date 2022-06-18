import jax.numpy as jnp
import numpy as np
import torch
import torchvision
from torch.utils import data
from torchvision import transforms


def load_data(resize=None):  #@save
    """Download the Fashion-MNIST dataset and then load it into memory."""
    trans = [transforms.ToTensor()]
    if resize: trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=True)
    X_train, X_test = mnist_train.data, mnist_test.data
    X_train, X_test = X_train.reshape(-1,28,28,1), X_test.reshape(-1,28,28,1)
    X_train, X_test = jnp.array(X_train), jnp.array(X_test)
    X_train, X_test = X_train/255.0, X_test/255.0
    Y_train, Y_test = mnist_train.targets, mnist_test.targets
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)
    classes =  np.unique(Y_train)
    class_labels = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    mapping = dict(zip(classes, class_labels))
    return X_train, X_test, Y_train, Y_test, classes, mapping
