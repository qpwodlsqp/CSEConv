#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adopted from https://github.com/antao97/PointCloudDatasets
MIT License

Copyright (c) 2019 An Tao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
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
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):

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
            self.root = os.path.join(root, 'dataset', dataset_name + '_hdf5_2048') # util -> dataset
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


