from .gaussian import GaussianORF, GaussianRFF
import torch

n,p,d=10,2,7
U = torch.randn(d,d)
M = U@(U.T) #Generate a random symmetric positive definite matrix

feature_map = GaussianORF(input_dim=d, num_features=p, shape_matrix=M)
DEVICE = feature_map.device
X = torch.randn(n,d).to(DEVICE)
Q = torch.randn(d,p).to(DEVICE)
S = torch.randn(p).to(DEVICE)
feature_map.set_Q(Q)
feature_map.set_S(S)
L = torch.linalg.cholesky(M).to(DEVICE)

Phi = feature_map.c1.to('cpu') * torch.cat([(torch.mm(X@L, Q)*S).cos(), (torch.mm(X@L, Q)*S).sin()], dim=-1)
if torch.allclose(feature_map(X), Phi, atol=1e-4):
    print("Gaussian ORF test complete")
    
feature_map = GaussianRFF(input_dim=d, num_features=p, shape_matrix=M)
DEVICE = feature_map.device
X = torch.randn(n,d).to(DEVICE)
W1 = torch.randn(d,p).to(DEVICE)
W2 = torch.randn(p).to(DEVICE)
feature_map.set_W1(W1)
feature_map.set_W2(W2) #Gaussian only requires W1 in RFF, W2 is not used

Phi = feature_map.c1.to('cpu') * torch.cat([(torch.mm(X@L, W1)).cos(), (torch.mm(X@L, W1)).sin()], dim=-1)
if torch.allclose(feature_map(X), Phi, atol=1e-4):
    print("Gaussian RFF test complete")
else:
    print(feature_map(X), Phi)