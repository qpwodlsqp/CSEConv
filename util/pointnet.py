import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):

    def __init__(self, num_points, c_in, c_out, h_dim=256):

        super().__init__()
        self.pointnet = nn.Sequential(
            nn.Conv2d(c_in, h_dim, 1),
            nn.ReLU(),
            nn.Conv2d(h_dim, c_out, 1),
            nn.ReLU(),
            nn.MaxPool2d((num_points, 1), 1)
        )
 
    def forward(self, x):
        # x: B x N x C => B x C x N x 1
        x = x.permute(0, 2, 1).unsqueeze(-1)
        # B x C x N x 1 => B x C x 1 x 1
        return self.pointnet(x).squeeze(-1).squeeze(-1)


