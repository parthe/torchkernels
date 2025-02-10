from torchvision.datasets import MNIST, EMNIST, KMNIST, CIFAR10
import os

_ = MNIST(os.environ['DATASETS_DIR'], train=True, download=True)
_ = EMNIST(os.environ['DATASETS_DIR'], split='digits', train=True, download=True)
_ = KMNIST(os.environ['DATASETS_DIR'], train=True, download=True)
_ = CIFAR10(os.environ['DATASETS_DIR'], train=True, download=True)
