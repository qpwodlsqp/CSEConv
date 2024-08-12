import torch

def sphere_coord_and_feature(X, r=1.):
    # Assume X is a set of \mathbb{R}^{3} coordinates
    # r is a dataset dependent value for feature normalization (KITTI; Velodyne HDL-64E)
    feature = torch.linalg.norm(X, dim=-1, keepdim=True)
    X = X / feature # unit vectors
    return X, feature/r 

def rotation_between_s2points(A, B):
    # A & B assumes (N, 3)
    # Return SO3 R which satisfies R @ A = B
    inner = (A * B).sum(dim=-1)
    cross = torch.nan_to_num(torch.linalg.cross(A, B, dim=-1), nan=0.0)
    C = cross_prod_mat(cross)
    scale = (1.-inner.view(-1, 1, 1)) / cross.norm(dim=-1).square().view(-1, 1, 1)
    scale = torch.nan_to_num(scale, nan=0.)
    R = torch.eye(3).to(C).unsqueeze(0) + C + torch.bmm(C, C) * scale
    return R
 
def cross_prod_mat(X):
    # Input: X \in R^{N x 3}
    # Output: [X]_{\times} \in R^{N x 3 x 3}
    I = torch.eye(3).to(X)
    I0 = I[0].repeat(X.size(0), 1)
    I1 = I[1].repeat(X.size(0), 1)
    I2 = I[2].repeat(X.size(0), 1)

    X_cross = torch.bmm(torch.cross(X, I0, dim=-1).unsqueeze(-1), I0.unsqueeze(1))\
            + torch.bmm(torch.cross(X, I1, dim=-1).unsqueeze(-1), I1.unsqueeze(1))\
            + torch.bmm(torch.cross(X, I2, dim=-1).unsqueeze(-1), I2.unsqueeze(1))
    if torch.isnan(X_cross).any():
        X_cross = torch.nan_to_num(X_cross, nan=0.0)
    return X_cross


