#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: An Tao
@Contact: ta19@mails.tsinghua.edu.cn
@File: dataset.py
@Time: 2020/1/2 10:26 AM
"""

import os
import torch
import json
import h5py
from glob import glob
import numpy as np
import torch.utils.data as data
import random
from pytorch3d.transforms import random_rotation

shapenetpart_cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
                       'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
                       'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
shapenetpart_seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
shapenetpart_seg_start_index = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[1, 3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[1, 3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32') 
    # translated_pointcloud = np.add(pointcloud, xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    '''
    Q, _ = np.linalg.qr(np.random.normal(size=(3, 3)))
    Q = Q.astype(np.float32)
    # theta = np.pi*2 * np.random.rand()
    # rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    # pointcloud[:,[0,1]] = pointcloud[:,[0,1]].dot(rotation_matrix) # random rotation (x,z)
    '''
    """ Randomly rotate the point clouds uniformly
        https://math.stackexchange.com/a/442423
        Input:
          BxNx3 array, point clouds
        Return:
          BxNx3 array, point clouds
    """
    '''
    rs = np.random.rand(3)
    angle_z1 = np.arccos(2 * rs[0] - 1)
    angle_y = np.pi*2 * rs[1]
    angle_z2 = np.pi*2 * rs[2]
    Rz1 = np.array([[np.cos(angle_z1),-np.sin(angle_z1),0],
                    [np.sin(angle_z1),np.cos(angle_z1),0],
                    [0,0,1]])
    Ry = np.array([[np.cos(angle_y),0,np.sin(angle_y)],
                    [0,1,0],
                    [-np.sin(angle_y),0,np.cos(angle_y)]])
    Rz2 = np.array([[np.cos(angle_z2),-np.sin(angle_z2),0],
                    [np.sin(angle_z2),np.cos(angle_z2),0],
                    [0,0,1]])
    R = np.dot(Rz2, np.dot(Ry, Rz1)).astype('float32')
    '''
    R = random_rotation().numpy()
    pointcloud = R.dot(pointcloud.T).T
    return pointcloud


class ModelNetDataset(data.Dataset):
    def __init__(self, root, dataset_name='modelnet40', class_choice=None,
            num_points=2048, split='train', load_name=True, load_file=False,
            segmentation=False, random_rotate=False, random_jitter=False, 
            random_translate=False, random_dropout=False):

        assert dataset_name.lower() in ['shapenetcorev2', 'shapenetpart', 
            'modelnet10', 'modelnet40', 'shapenetpartpart']
        assert num_points <= 2048        

        if dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            assert split.lower() in ['train', 'test', 'val', 'trainval', 'all']
        else:
            assert split.lower() in ['train', 'test', 'all']

        if dataset_name not in ['shapenetpart'] and segmentation == True:
            raise AssertionError

        if os.path.exists(os.path.join(root, dataset_name + '_hdf5_2048')):
            self.root = os.path.join(root, dataset_name + '_hdf5_2048')
        else:
            self.root = os.path.join(root, 'util', dataset_name + '_hdf5_2048')
        self.dataset_name = dataset_name
        self.class_choice = class_choice
        self.num_points = num_points
        self.split = split
        self.load_name = load_name
        self.load_file = load_file
        self.segmentation = segmentation
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.random_dropout = random_dropout

        self.path_h5py_all = []
        self.path_name_all = []
        self.path_file_all = []

        if self.split in ['train', 'trainval', 'all']:   
            self.get_path('train')
        if self.dataset_name in ['shapenetcorev2', 'shapenetpart', 'shapenetpartpart']:
            if self.split in ['val', 'trainval', 'all']: 
                self.get_path('val')
        if self.split in ['test', 'all']:   
            self.get_path('test')

        data, label, seg = self.load_h5py(self.path_h5py_all)

        if self.load_name or self.class_choice != None:
            self.name = np.array(self.load_json(self.path_name_all))    # load label name

        if self.load_file:
            self.file = np.array(self.load_json(self.path_file_all))    # load file name
        
        self.data = np.concatenate(data, axis=0)
        self.label = np.concatenate(label, axis=0) 
        if self.segmentation:
            self.seg = np.concatenate(seg, axis=0) 

        if self.class_choice != None:
            indices = (self.name == class_choice)
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.name = self.name[indices]
            if self.segmentation:
                self.seg = self.seg[indices]
                id_choice = shapenetpart_cat2id[class_choice]
                self.seg_num_all = shapenetpart_seg_num[id_choice]
                self.seg_start_index = shapenetpart_seg_start_index[id_choice]
            if self.load_file:
                self.file = self.file[indices]
        elif self.segmentation:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def get_path(self, type):
        path_h5py = os.path.join(self.root, '%s*.h5'%type)
        paths = glob(path_h5py)
        paths_sort = [os.path.join(self.root, type + str(i) + '.h5') for i in range(len(paths))]
        self.path_h5py_all += paths_sort
        if self.load_name:
            paths_json = [os.path.join(self.root, type + str(i) + '_id2name.json') for i in range(len(paths))]
            self.path_name_all += paths_json
        if self.load_file:
            paths_json = [os.path.join(self.root, type + str(i) + '_id2file.json') for i in range(len(paths))]
            self.path_file_all += paths_json
        return 

    def load_h5py(self, path):
        all_data = []
        all_label = []
        all_seg = []
        for h5_name in path:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            if self.segmentation:
                seg = f['seg'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            if self.segmentation:
                all_seg.append(seg)
        return all_data, all_label, all_seg

    def load_json(self, path):
        all_data = []
        for json_name in path:
            j =  open(json_name, 'r+')
            data = json.load(j)
            all_data += data
        return all_data

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        # point_set = self.data[item]
        # # convert numpy array to pytorch Tensor
        # if self.split in ['train', 'trainval', 'all']:   
        #     np.random.shuffle(point_set)
        # point_set = point_set[:self.num_points]
        label = self.label[item]
        if self.load_name:
            name = self.name[item]  # get label name
        if self.load_file:
            file = self.file[item]  # get file name

        point_set = point_set - point_set.mean(axis=0, keepdims=True)
        point_set = point_set / np.sqrt((point_set**2).sum(axis=-1, keepdims=True)).max()
 
        if self.random_jitter:
            point_set = jitter_pointcloud(point_set)
        if self.random_translate:
            point_set = translate_pointcloud(point_set)
        if self.random_dropout:
            point_set = random_point_dropout(point_set)
        if self.random_rotate:
            point_set = rotate_pointcloud(point_set)
 
        # convert numpy array to pytorch Tensor
        if self.split in ['train', 'trainval', 'all']:   
            np.random.shuffle(point_set)
        point_set = torch.from_numpy(point_set)
        label = torch.from_numpy(np.array([label]).astype(np.int64))
        label = label.squeeze(0)
        
        if self.segmentation:
            seg = self.seg[item]
            seg = torch.from_numpy(seg)
            return point_set, label , seg # , name, file
        else:
            return point_set, label # , name, file

    def __len__(self):
        return self.data.shape[0]


class ModelNet:

    def __init__(self, num_points=2048, use_rotate=False, use_noisy=False, test=False, batch=16, workers=0):

        root = os.getcwd()
        # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
        dataset_name = 'modelnet40'
        # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
        # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
        split = 'test' if test else 'train'
        self.dataset = ModelNetDataset(root=root,
                                       dataset_name = dataset_name,
                                       num_points = num_points,
                                       split = split,
                                       random_rotate = use_rotate,
                                       random_jitter = use_noisy,
                                       random_translate = use_noisy,
                                       random_dropout = use_noisy)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
 
        self.dataloader = data.DataLoader(self.dataset,
                                          batch_size = batch,
                                          shuffle = not test,
                                          num_workers = workers,
                                          worker_init_fn = seed_worker,
                                          generator = g)
        return


if __name__ == '__main__':
    from pytorch3d.ops import estimate_pointcloud_local_coord_frames
    from pytorch3d.transforms import random_rotation
    import time
    root = os.getcwd()
    class_choice = ['airplane', 'bathtub', 'bed', 'bench','bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone','cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
 
    # choose dataset name from 'shapenetcorev2', 'shapenetpart', 'modelnet40' and 'modelnet10'
    dataset_name = 'modelnet40'

    # choose split type from 'train', 'test', 'all', 'trainval' and 'val'
    # only shapenetcorev2 and shapenetpart dataset support 'trainval' and 'val'
    split = 'train'
    device = torch.device('cuda')
    modelnet = ModelNet(test=False, batch=1)
    print(f"datasize: {len(modelnet.dataset)}")
    '''
    for i, (pc, label) in enumerate(modelnet.dataloader):
        st = time.time()
        pc = pc.to(device)
        # _, F = sphere_coord_and_feature_exp(pc)
        curvatures = estimate_pointcloud_local_coord_frames(pc, neighborhood_size=32)
        org_curvs = curvatures[1]
        org_inner1 = (org_curvs[:, :, :, 0] * org_curvs[:, :, :, 1]).sum(dim=-1)
        org_inner2 = (org_curvs[:, :, :, 1] * org_curvs[:, :, :, 2]).sum(dim=-1)
        org_inner3 = (org_curvs[:, :, :, 2] * org_curvs[:, :, :, 0]).sum(dim=-1)
        for j in range(5):
            R = random_rotation(device=device)
            pc = pc @ R.T
            curvatures = estimate_pointcloud_local_coord_frames(pc, neighborhood_size=32)
            curvs = curvatures[1]
            # _, Fr = sphere_coord_and_feature_exp(pc)
            # diff = torch.abs(Fr[:, :, 1:] - F[:, :, 1:])
            inner1 = (curvs[:, :, :, 0] * curvs[:, :, :, 1]).sum(dim=-1)
            inner2 = (curvs[:, :, :, 1] * curvs[:, :, :, 2]).sum(dim=-1)
            inner3 = (curvs[:, :, :, 2] * curvs[:, :, :, 0]).sum(dim=-1)
            print(torch.allclose(org_inner1, inner1, atol=1e-5))
            print(torch.allclose(org_inner2, inner2, atol=1e-5))
            print(torch.allclose(org_inner3, inner3, atol=1e-5))
            # print(diff.mean())
            # print(diff.std())
            # print(torch.allclose(F, Fr, rtol=0., atol=1e-3))
        ed = time.time()
        print(ed-st)
        if i >=5: break
    '''
    cls_nums = np.zeros(40)
    for i, (pc, label) in enumerate(modelnet.dataloader):
 
        '''
        pc = pc.squeeze(0)
        print(pc.mean(dim=0))
        print(pc.norm(dim=-1).max())
        if i >= 9: break
        '''
        cls_nums[label.item()] += 1
    print(cls_nums)
    print(class_choice)
    print(cls_nums[class_choice.index('flower_pot')])
    print(cls_nums[class_choice.index('vase')])
    print(cls_nums[class_choice.index('plant')])


