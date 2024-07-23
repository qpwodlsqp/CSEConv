import torch
import torch.nn as nn
import pytorch3d.ops as ops
import pytorch3d.transforms as trans
import numpy as np

from networks.layers import *
from util.pointnet import PointNet
from util.geo import *

class SO3ModelNet(nn.Module):

    def __init__(self, cfg_dict, class_dim=40):

        super().__init__()
        conv_len = len(cfg_dict['conv'])
        conv_layers = []
        bn_layers = []
        for i in range(conv_len):
            layer_dict = cfg_dict['conv'][i]
            conv_layer = SO3EquivConv(input_dim=layer_dict['input_dim'],
                                      output_dim=layer_dict['output_dim'],
                                      fourier_dim=layer_dict['fourier_dim'],
                                      hidden_dim=layer_dict['hidden_dim'],
                                      hidden_num=layer_dict['hidden_num'],
                                      freq=layer_dict['freq'],
                                      do_norm=layer_dict['do_norm'],
                                      do_sample=layer_dict['do_sample'],
                                      k=layer_dict['k'],
                                      mc_num=layer_dict['mc_num'],
                                      activation = layer_dict['activation'])
            conv_layers.append(conv_layer)
            bn_layers.append(nn.BatchNorm1d(layer_dict['output_dim']))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers   = nn.ModuleList(bn_layers)

        self.pointnet = PointNet(cfg_dict['conv'][conv_len-1]['mc_num'],
                                 cfg_dict['conv'][conv_len-1]['output_dim'],
                                 1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), # 64
            nn.ReLU(),
            nn.Linear(256, class_dim),
        )


    def forward(self, X):
        # Before convolution, change them into coordinates on 2-sphere
        # and feature (length from origin to coordinate)
        X = X - X.mean(dim=1, keepdim=True)
        _, F = sphere_coord_and_feature(X)
        # _, F = sphere_coord_and_feature_exp(X)
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            X, F = conv_layer(X, F)
            F = bn_layer(F.permute(0, 2, 1)).permute(0, 2, 1)
        F = self.pointnet(F)
        F = self.classifier(F) # new
        # return nn.functional.log_softmax(F, dim=-1)
        return F


class SO3ModelNet2(nn.Module):

    def __init__(self, cfg_dict, class_dim=40):

        super().__init__()
        conv_len = len(cfg_dict['conv'])
        conv_layers = []
        bn_layers = []
        for i in range(conv_len):
            layer_dict = cfg_dict['conv'][i]
            conv_layer = SO3EquivConv2(input_dim=layer_dict['input_dim'],
                                       output_dim=layer_dict['output_dim'],
                                       fourier_dim=layer_dict['fourier_dim'],
                                       hidden_dim=layer_dict['hidden_dim'],
                                       hidden_num=layer_dict['hidden_num'],
                                       freq=layer_dict['freq'],
                                       do_sample=layer_dict['do_sample'],
                                       k=layer_dict['k'],
                                       mc_num=layer_dict['mc_num'],
                                       activation = layer_dict['activation'])
            conv_layers.append(conv_layer)
            bn_layers.append(nn.BatchNorm1d(layer_dict['output_dim']))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers   = nn.ModuleList(bn_layers)

        self.pointnet = PointNet(cfg_dict['conv'][conv_len-1]['mc_num'],
                                 cfg_dict['conv'][conv_len-1]['output_dim'],
                                 1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), # 64
            nn.ReLU(),
            nn.Linear(256, class_dim),
        )


    def forward(self, X):
        # Before convolution, change them into coordinates on 2-sphere
        # and feature (length from origin to coordinate)
        _, F = sphere_coord_and_feature(X)
        # _, F = sphere_coord_and_feature_exp(X)
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            X, F = conv_layer(X, F)
            F = bn_layer(F.permute(0, 2, 1)).permute(0, 2, 1)
        F = self.pointnet(F)
        F = self.classifier(F) # new
        # return nn.functional.log_softmax(F, dim=-1)
        return F


class SO3ScanNN(nn.Module):

    def __init__(self, cfg_dict, class_dim=15):

        super().__init__()
        conv_len = len(cfg_dict['conv'])
        conv_layers = []
        bn_layers = []
        for i in range(conv_len):
            layer_dict = cfg_dict['conv'][i]
            conv_layer = SO3EquivConv(input_dim=layer_dict['input_dim'],
                                      output_dim=layer_dict['output_dim'],
                                      fourier_dim=layer_dict['fourier_dim'],
                                      hidden_dim=layer_dict['hidden_dim'],
                                      hidden_num=layer_dict['hidden_num'],
                                      freq=layer_dict['freq'],
                                      do_norm=layer_dict['do_norm'],
                                      do_sample=layer_dict['do_sample'],
                                      k=layer_dict['k'],
                                      mc_num=layer_dict['mc_num'],
                                      activation = layer_dict['activation'])
            conv_layers.append(conv_layer)
            bn_layers.append(nn.BatchNorm1d(layer_dict['output_dim']))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers   = nn.ModuleList(bn_layers)

        self.pointnet = PointNet(cfg_dict['conv'][conv_len-1]['mc_num'],
                                 cfg_dict['conv'][conv_len-1]['output_dim'],
                                 1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256), # 64
            nn.ReLU(),
            nn.Linear(256, class_dim),
        )

    def forward(self, X):
        # Before convolution, change them into coordinates on 2-sphere
        # and feature (length from origin to coordinate)
        # _, F = sphere_coord_and_feature_exp(X)
        _, F = sphere_coord_and_feature(X)
        # _, F = sphere_coord_and_feature_exp(X)
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            X, F = conv_layer(X, F)
            F = bn_layer(F.permute(0, 2, 1)).permute(0, 2, 1)
        F = self.pointnet(F)
        F = self.classifier(F) # new
        return nn.functional.log_softmax(F, dim=-1)


class SO3ModelNetMetric(nn.Module):

    def __init__(self, cfg_dict):

        super().__init__()
        conv_len = len(cfg_dict['conv'])
        conv_layers = []
        bn_layers = []
        self.metric_dim = 256
        for i in range(conv_len):
            layer_dict = cfg_dict['conv'][i]
            conv_layer = SO3EquivConv(input_dim=layer_dict['input_dim'],
                                      output_dim=layer_dict['output_dim'],
                                      fourier_dim=layer_dict['fourier_dim'],
                                      hidden_dim=layer_dict['hidden_dim'],
                                      hidden_num=layer_dict['hidden_num'],
                                      freq=layer_dict['freq'],
                                      do_norm=layer_dict['do_norm'],
                                      do_sample=layer_dict['do_sample'],
                                      k=layer_dict['k'],
                                      mc_num=layer_dict['mc_num'],
                                      activation = layer_dict['activation'])
            conv_layers.append(conv_layer)
            bn_layers.append(nn.BatchNorm1d(layer_dict['output_dim']))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.bn_layers   = nn.ModuleList(bn_layers)
        self.pointnet = PointNet(cfg_dict['conv'][conv_len-1]['mc_num'],
                                 cfg_dict['conv'][conv_len-1]['output_dim'],
                                 1024)
        self.metric = nn.Sequential(
            nn.Linear(1024, 2048), # 64
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.metric_dim),
        )


    def get_feature(self, X):
        # Only for visualization purpose
        _, F = sphere_coord_and_feature(X)
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            X, F = conv_layer(X, F)
            F = bn_layer(F.permute(0, 2, 1)).permute(0, 2, 1)
        F_pre = self.pointnet(F)
        F_met = self.metric(F_pre)
        # return F
        return F_pre, nn.functional.normalize(F_met, p=2, dim=-1)

    def forward(self, X):
        # Before convolution, change them into coordinates on 2-sphere
        # and feature (length from origin to coordinate)
        _, F = sphere_coord_and_feature(X)
        for conv_layer, bn_layer in zip(self.conv_layers, self.bn_layers):
            X, F = conv_layer(X, F)
            F = bn_layer(F.permute(0, 2, 1)).permute(0, 2, 1)
        F = self.pointnet(F)
        F = self.metric(F)
        # return F
        return nn.functional.normalize(F, p=2, dim=-1)


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True
    import time
    #from util.oxford import Oxford
    from util.shapenetpart import ShapeNetPart
    from omegaconf import OmegaConf
    import pytorch3d
    from torch.profiler import profile, record_function, ProfilerActivity
    device = torch.device('cuda')
    cfg_dict = OmegaConf.load('cfg/shapenetpart/shapenetpart_default.yaml')
    model = CSE2ConvShapeNetPart(cfg_dict)
    model = model.to(device)
    oxford = ShapeNetPart(num_points=2048, split='train', batch=8)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof: 
        for i, (pc, label, seg) in enumerate(oxford.dataloader):
            pc = pc.to(device); label = label.to(device)
            f = model(pc, label)
            print(f.shape); print(seg.shape)
            if i >= 25: break
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))


