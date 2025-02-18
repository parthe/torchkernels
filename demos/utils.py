import torchvision
import torchvision.transforms as transforms
import os

def load_fmnist(device='cpu'):
  transform = transforms.Compose([transforms.ToTensor()])
  root_dir = os.environ.get("DATASETS_DIR", "./data")
  train_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=True, transform=transform, download=True)
  test_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=False, transform=transform, download=True)
  
  X_train = train_dataset.data.to(device).flatten(1).float()/255.0
  X_test = test_dataset.data.to(device).flatten(1).float()/255.0
  normalize = lambda x : x/x.norm(dim=-1, keepdim=True)
  X_train, X_test = normalize(X_train), normalize(X_test)
  y_train = train_dataset.targets.to(device)
  y_test = test_dataset.targets.to(device)
  
  return X_train, X_test, y_train, y_test
