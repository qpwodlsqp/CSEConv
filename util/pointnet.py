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

class PointNetBN(nn.Module):

    def __init__(self, num_points, c_in, c_out, h_dim=256, pool='max'):

        super().__init__()
        layers = [
            nn.Conv1d(c_in, h_dim, 1),
            nn.BatchNorm1d(h_dim),
            nn.GELU(),
            nn.Conv1d(h_dim, c_out, 1),
            nn.BatchNorm1d(c_out),
            nn.GELU()
        ]
        if pool == 'max':
            layers.append(nn.MaxPool1d(num_points))
        elif pool == 'mean':
            layers.append(nn.AvgPool1d(num_points))
        else:
            raise NotImplementedError
        self.pointnet = nn.Sequential(*layers)
 
    def forward(self, x):
        # x: B x N x C => B x C x N 
        x = x.permute(0, 2, 1)
        # B x C x N => B x C
        return self.pointnet(x).squeeze(-1)


