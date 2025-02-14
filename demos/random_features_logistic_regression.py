import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torchkernels.feature_maps import LaplacianRFF
from torchmetrics.classification import Accuracy, CalibrationError

class LogisticRegressionRidge(nn.Module):
    def __init__(self, input_dim, lambda_):
        super(LogisticRegressionRidge, self).__init__()
        self.linear = nn.Linear(input_dim, 10)  # 10 classes for FMNIST
        self.lambda_ = lambda_
    
    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)
    
    def compute_loss(self, y_pred, y_true):
        criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class
        base_loss = criterion(y_pred, y_true)
        ridge_penalty = self.lambda_ * torch.sum(self.linear.weight ** 2)  # L2 Regularization
        return base_loss + ridge_penalty

# Load FMNIST dataset
def load_fmnist(device='cpu'):
    transform = transforms.Compose([transforms.ToTensor()])
    root_dir = os.environ.get("DATASETS_DIR", "./data")
    train_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=True, transform=transform, download=True)
    test_dataset = torchvision.datasets.FashionMNIST(root=root_dir, train=False, transform=transform, download=True)
    
    X_train = train_dataset.data.float().view(train_dataset.data.size(0), -1).to(device) / 255.0
    X_test = test_dataset.data.float().view(test_dataset.data.size(0), -1).to(device) / 255.0
    def normalize(X): return X/X.norm(dim=-1, keepdim=True)
    X_train, X_test = normalize(X_train), normalize(X_test)
    y_train = train_dataset.targets.to(device)
    y_test = test_dataset.targets.to(device)
    
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, input_dim, lambda_, batch_size=2048, lr=0.01, epochs=100, device='cpu'):
    model = LogisticRegressionRidge(input_dim, lambda_).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    num_samples = X_train.size(0)
    for epoch in range(epochs):
        total_loss = 0.0
        perm = torch.randperm(num_samples)  # Shuffle data each epoch
        
        for i in range(0, num_samples, batch_size):
            batch_indices = perm[i:i + batch_size]
            X_batch, y_batch = X_train[batch_indices], y_train[batch_indices]
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = model.compute_loss(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (num_samples // batch_size)
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')    
    return model

# Compute accuracy
def compute_accuracy(model, X, y, batch_size=64, device='cpu'):
    accuracy_metric = Accuracy(task='multiclass', num_classes=10).to(device)
    num_samples = X.size(0)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            X_batch, y_batch = X[i:i + batch_size], y[i:i + batch_size]
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            accuracy_metric.update(predicted, y_batch)
    
    return accuracy_metric.compute().item()

# Compute Expected Calibration Error (ECE) using torchmetrics
def compute_ece(model, X, y, batch_size=64, device='cpu'):
    ece_metric = CalibrationError(task='multiclass', num_classes=10).to(device)
    num_samples = X.size(0)
    
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            X_batch, y_batch = X[i:i + batch_size], y[i:i + batch_size]
            probs = model(X_batch)
            ece_metric.update(probs, y_batch)
    
    return ece_metric.compute().item()

# torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = 28 * 28  # FMNIST images are 28x28
lambda_ = 2e-6  # Regularization strength
num_features = 10000

X_train, X_test, y_train, y_test = load_fmnist(device=DEVICE)
feature_map = LaplacianRFF(input_dim=input_dim, num_features=num_features)
X_train = feature_map(X_train)
X_test = feature_map(X_test)

trained_model = train_model(X_train, y_train, num_features, lambda_, device=DEVICE)

# Compute accuracies
train_accuracy = compute_accuracy(trained_model, X_train, y_train, device=DEVICE)
test_accuracy = compute_accuracy(trained_model, X_test, y_test, device=DEVICE)

# Compute ECE
train_ece = compute_ece(trained_model, X_train, y_train, device=DEVICE)
test_ece = compute_ece(trained_model, X_test, y_test, device=DEVICE)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Train ECE: {train_ece:.4f}")
print(f"Test ECE: {test_ece:.4f}")