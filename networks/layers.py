import torch
import torch.nn as nn
import pytorch3d.ops as ops
import pytorch3d.transforms as trans
import numpy as np
from util.geo import *

class SO3EquivConvBlock(nn.Module):

    def __init__(self, in_dim, out_dim, fourier_dim=16, hidden_dim=128, hidden_num=2, freq=16, do_norm=False):

        super().__init__()
        # Originally 24*pi 8pi for ScanObjectNN
        self.W = nn.parameter.Parameter(nn.init.normal_(torch.empty(1, fourier_dim)), requires_grad=False)

        filter_layers = [nn.Linear(fourier_dim*2, hidden_dim)]
        for i in range(hidden_num):
            filter_layers.append(nn.GELU())  # ReLU
            if i < hidden_num - 1:
                filter_layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                filter_layers.append(nn.Linear(hidden_dim, out_dim*in_dim))
        self.filter = nn.Sequential(*filter_layers)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fourier_dim = fourier_dim
        self.do_norm = do_norm
        self.freq = freq

    def to_spherical(self, X, k):
        # Input:  X \in R^{B x (#query x k) x 3} coordinates
        # Output: X \in R^{B x (#query x k) x 3} which represents the displacement in the S^2 space \equiv so(3)
        B = X.size(0); N = X.size(1)
        X = X.view(B, N//k, k, 3)       # R^{B x #query x k x 3}
        Q = X[:, :, 0, :].unsqueeze(-2) # R^{B x #query x 1 x 3} which are the centroids of bundles
        Q = Q.expand(B, N//k, k, 3).reshape(B*N, 3)
        npole = torch.zeros_like(Q); npole[:, 2] += 1
        R_ref = rotation_between_s2points(Q, npole)
        X = torch.bmm(R_ref, X.view(-1, 3, 1))
        X = X.view(B, N, 3)
        theta = torch.acos(torch.clip(X[:, :, 2:3], min=-1.+1e-6, max=1.-1e-6))
        # phi   = torch.atan2(X[:, :, 1:2], X[:, :, 0:1])
        return theta

    def to_fourier_feature(self, X):

        if self.do_norm:
            X = X @ (self.W / self.W.norm(p='fro') * torch.pi * self.freq)
        else:
            X = X @ (self.W * torch.pi * self.freq)
        return torch.cat([torch.sin(X), torch.cos(X)], dim=-1)

    def forward(self, X, F, k=1):
        # Input:  X \in R^{B x (#mc-sample x #query x k) x 3} coordinates
        #         F \in R^{B x (#mc-sample x #query x k) x D} features
        #         k is KNN parameter
        # Output: (X * Filter) \in R^{B x #query x 1}
        B = X.size(0); N = X.size(1)
        X = self.to_spherical(X, k) # B x (#query x k) x 1 theta from spherical coordinates s(q)^{-1} \cdot x
        # [B x (#query x k) x 1] -> [B x (#query x k) x (2fourier_dim)]
        X = self.to_fourier_feature(X)
        kernel = self.filter(X).view(B, N//k, k, self.out_dim, self.in_dim)
        F = F.view(B, N//k, k, self.in_dim).unsqueeze(-2).expand(B, N//k, k, self.out_dim, self.in_dim)
        convolution = (kernel * F).sum(dim=-1).mean(dim=-2)
        return convolution # B x #query x D_out


class SO3EquivConv(nn.Module):

    def __init__(self, input_dim, output_dim, fourier_dim, hidden_dim, hidden_num, freq, do_norm,
                 do_sample=False, k=16, mc_num=4, activation='relu'):

        super().__init__()
        self.conv_layer = SO3EquivConvBlock(input_dim, output_dim,
                                            fourier_dim, hidden_dim,
                                            hidden_num, freq, do_norm)
        self.k = k # KNN parameter
        self.mc_num = mc_num
        self.do_sample = do_sample
        self.activation = getattr(nn.functional, activation)

    def forward(self, X, F):

        B = X.size(0); N = X.size(1); D = F.size(-1)
        # old do sample logic
        do_sample = self.do_sample if self.training else False
        Q, q_idx = ops.sample_farthest_points(X, K=self.mc_num, random_start_point=do_sample)
        # KNN neighbor
        knn_result = ops.knn_points(Q, X, K=self.k, return_nn=True)
        knn_X = knn_result.knn.view(B, -1, 3)
        knn_X = knn_X / knn_X.norm(dim=-1, keepdim=True) # NEW !!!!
        knn_F = ops.knn_gather(F, knn_result.idx.clone()).view(B, -1, F.size(-1))
        conv_F = self.conv_layer(knn_X, knn_F, self.k)
        return Q, self.activation(conv_F)


