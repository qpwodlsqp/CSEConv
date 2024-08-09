import torch
from pytorch3d.ops import estimate_pointcloud_normals, estimate_pointcloud_local_coord_frames

def sphere_coord_and_feature_move(X, r=1.):
    # Assume X is a set of \mathbb{R}^{3} coordinates
    # r is a dataset dependent value for feature normalization (KITTI; Velodyne HDL-64E)
    center = X.mean(dim=-2, keepdim=True)
    X = X - center
    feature = torch.linalg.norm(X, dim=-1, keepdim=True)
    X = X / feature # unit vectors
    return X, feature/r 

def sphere_coord_and_feature(X, r=1.):
    # Assume X is a set of \mathbb{R}^{3} coordinates
    # r is a dataset dependent value for feature normalization (KITTI; Velodyne HDL-64E)
    feature = torch.linalg.norm(X, dim=-1, keepdim=True)
    X = X / feature # unit vectors
    return X, feature/r 

def sphere_coord_and_feature_exp(X):
    # Assume X is a set of \mathbb{R}^{3} coordinates
    # r is a dataset dependent value for feature normalization (KITTI; Velodyne HDL-64E)
    norm = torch.linalg.norm(X, dim=-1, keepdim=True)
    coord = X / norm # unit vectors
    normals = estimate_pointcloud_normals(X, neighborhood_size=20)
    normals = torch.nan_to_num(normals, nan=0.)
    cos = (coord * normals).sum(dim=-1, keepdim=True)
    sin = (1. - cos.square()).sqrt()
    return coord, torch.cat((norm, cos, sin), dim=-1)

def surface_feature(X):
    curvs = estimate_pointcloud_local_coord_frames(X, neighborhood_size=32)[1]
    curvs = torch.nan_to_num(curvs, nan=0.)
    angle0 = (curvs[:, :, :, 0] * curvs[:, :, :, 1]).sum(dim=-1).arccos()
    angle1 = (curvs[:, :, :, 1] * curvs[:, :, :, 2]).sum(dim=-1).arccos()
    angle2 = (curvs[:, :, :, 2] * curvs[:, :, :, 0]).sum(dim=-1).arccos()
    return torch.stack((angle0, angle1, angle2), dim=-1)

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

def sphere_to_so3_algebra_vector(a, b):
    # Input: a \in R^{N x 3}, b \in R^{M x 3}, assume a & b are coordinates on 2-sphere
    # Output: batches of h \in so(3)^{\vee} [N x M x 3] shape
    theta = torch.acos(torch.clamp(torch.einsum('ni, mi -> nm', a, b), -1. + 1e-6, 1. - 1e-6)).unsqueeze(-1) # N x M x 1
    N = a.size(0); M = b.size(0)
    a = a.unsqueeze(1).repeat(1, M, 1).view(N*M, 3)
    b = b.repeat(N, 1)
    normal = torch.nan_to_num(torch.cross(a, b, dim=-1), nan=0.0).view(N, M, 3)
    return theta * normal

def group_to_so3_algebra_vector(a, b):
    # Input: a \in R^{N x 3}, b \in R^{M x 3}, assume a & b are actually coordinates of so3 algebra
    # Output: batches of h \in so(3)^{\vee} [N x M x 3] shape
    A = torch.linalg.matrix_exp(cross_prod_mat(a)) # N x 3 x 3
    B = torch.linalg.matrix_exp(cross_prod_mat(b)) # M x 3 x 3
    R = torch.einsum('nij, mjk -> nmik', A.permute(0, 2, 1), B) # N x M x 3 x 3
    C = 0.5 * (R - R.permute(0, 1, 3, 2))
    C_sq = torch.einsum('nmij, nmjk -> nmik', C, C)
    norm = torch.sqrt(-0.5 * torch.einsum('nmii -> nm', C_sq)).unsqueeze(-1).unsqueeze(-1)
    logR = torch.nan_to_num(torch.arcsin(norm) / norm, nan=1.0) * C # N x M x 3 x 3, skew matrix of log R
    so3 = torch.stack([logR[:, :, 2, 1], logR[:, :, 0, 2], logR[:, :, 1, 0]], dim=-1) # N x M x 3, s03 coordinate vector
    return so3

def group_to_so3_algebra_vector_Z(z):
    # Input: z \in R^{D x P x 3}, assume z is actually coordinates of so3 algebra
    # Output: batches of h \in so(3)^{\vee} [D x P x P x 3] shape
    D = z.size(0); P = z.size(1)
    Z = torch.linalg.matrix_exp(cross_prod_mat(z.view(-1, 3))) # (D x P) x 3 x 3
    Z = Z.view(D, P, 3, 3)
    R = torch.einsum('abij, acjk -> abcik', Z.permute(0, 1, 3, 2), Z) # D x P x P x 3 x 3
    C = 0.5 * (R - R.permute(0, 1, 2, 4, 3))
    C_sq = torch.einsum('abcij, abcjk -> abcik', C, C)
    norm = torch.sqrt(-0.5 * torch.einsum('abcii -> abc', C_sq)).unsqueeze(-1).unsqueeze(-1) # D x P x P x 1 x 1
    logR = torch.nan_to_num(torch.arcsin(norm) / norm, nan=1.0) * C # D x P x P x 3 x 3, skew matrix of log R
    so3 = torch.stack([logR[:, :, :, 2, 1], logR[:, :, :, 0, 2], logR[:, :, :, 1, 0]], dim=-1)
    # D x P x P x 3, s03 coordinate vector
    return so3

def periodic_basis(x, omega):
    # Input: x \in R^{... x 3}, omega \in R^{3}
    # Output: x \in R^{...} (last dimension is scalar and squeezed)
    omega_sq = omega.square().unsqueeze(0).unsqueeze(0)
    return (-2. * omega_sq * torch.pow(torch.sin(torch.abs(x) * torch.pi * 0.5), 2)).sum(dim=-1).exp()

def wavelet_basis(x, omega):
    # Input: x \in R^{... x 3}, omega \in R^{3}
    # Output: x \in R^{...} (last dimension is scalar and squeezed)
    omega = omega.unsqueeze(0).unsqueeze(0)
    x = x / (omega + 1e-6)
    h = torch.cos(1.75 * x) * torch.exp(-0.5 * x.square())
    return h.prod(dim=-1)

def rbf_basis(x, omega):
    # Input: x \in R^{... x 3}, omega \in R^{3}
    # Output: x \in R^{...} (last dimension is scalar and squeezed)
    omega_sq = omega.square().unsqueeze(0).unsqueeze(0)
    return (-0.5 * omega_sq * torch.pow(x, 2)).sum(dim=-1).exp()

if __name__ == '__main__':

    X = torch.randn(200, 3)
    X = X / X.norm(dim=-1, keepdim=True)
    #X = torch.zeros(5, 3); X[:, 0] += 1
    Y = torch.randn(16, 3)
    Y = Y / Y.norm(dim=-1, keepdim=True)
    #Y = torch.zeros(5, 3); Y[:, 0] += -1
    so3 = so3_algebra_vector(X, Y)
    print(so3)
    '''
    n = torch.cross(X, Y, dim=-1)
    n_ = n / n.norm(dim=-1, keepdim=True)
    theta = torch.acos(torch.bmm(X.unsqueeze(1), Y.unsqueeze(-1)))

    K = cross_prod_mat(n_)
    print(X)
    print(Y)
    print(theta)
    print(K)
    thetaK = theta * K
    R = torch.linalg.matrix_exp(thetaK)
    print(R)
    print(torch.bmm(R, R.permute(0, 2, 1)))
    print(torch.bmm(R.permute(0, 2, 1), R))
    '''
