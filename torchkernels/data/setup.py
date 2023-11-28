from torchvision.datasets import MNIST, EMNIST, KMNIST, CIFAR10
import os

_ = MNIST(os.environ['DATA_DIR'], train=True, download=True)
_ = EMNIST(os.environ['DATA_DIR'], split='digits', train=True, download=True)
_ = KMNIST(os.environ['DATA_DIR'], train=True, download=True)
_ = CIFAR10(os.environ['DATA_DIR'], train=True, download=True)
