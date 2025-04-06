from .laplacian import LaplacianORF, LaplacianRFF, LaplacianORF_QMC
import torch

n,p,d=10,2,7
U = torch.randn(d,d)
M = U@(U.T) #Generate a random symmetric positive definite matrix

feature_map = LaplacianORF(input_dim=d, num_features=p, shape_matrix=M)
DEVICE = feature_map.device
X = torch.randn(n,d).to(DEVICE)
Q = torch.randn(d,p).to(DEVICE)
S = torch.randn(p).to(DEVICE)
feature_map.set_Q(Q)
feature_map.set_S(S)
L = torch.linalg.cholesky(M).to(DEVICE)

Phi = feature_map.c1.to('cpu') * torch.cat([(torch.mm(X@L, Q)*S).cos(), (torch.mm(X@L, Q)*S).sin()], dim=-1)
if torch.allclose(feature_map(X), Phi, atol=1e-4):
    print("Laplace ORF test complete")
    
feature_map = LaplacianORF_QMC(input_dim=d, num_features=p, shape_matrix=M)
DEVICE = feature_map.device
X = torch.randn(n,d).to(DEVICE)
Q = torch.randn(d,p).to(DEVICE)
S = torch.randn(p).to(DEVICE)
feature_map.set_Q(Q)
feature_map.set_S(S)
L = torch.linalg.cholesky(M).to(DEVICE)

Phi = feature_map.c1.to('cpu') * torch.cat([(torch.mm(X@L, Q)*S).cos(), (torch.mm(X@L, Q)*S).sin()], dim=-1)
if torch.allclose(feature_map(X), Phi, atol=1e-4):
    print("Laplace ORF_QMC test complete")
    
feature_map = LaplacianRFF(input_dim=d, num_features=p, shape_matrix=M)
DEVICE = feature_map.device
X = torch.randn(n,d).to(DEVICE)
W1 = torch.randn(d,p).to(DEVICE)
W2 = torch.randn(p).to(DEVICE)
feature_map.set_W1(W1)
feature_map.set_W2(W2)

Phi = feature_map.c1.to('cpu') * torch.cat([(torch.mm(X@L, W1)/W2).cos(), (torch.mm(X@L, W1)/W2).sin()], dim=-1)
if torch.allclose(feature_map(X), Phi, atol=1e-4):
    print("Laplace RFF test complete")