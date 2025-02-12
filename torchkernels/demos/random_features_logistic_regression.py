from torchkernels.feature_maps import LaplacianRFF
from torchvision.datasets import FMNIST
import os

data = FMNIST(os.environ['DATASETS_DIR'])

feature_map = LaplacianRFF()
Phi = feature_map(X)

# code for logistic regression