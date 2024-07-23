"""
code adopted from: https://github.com/ma-xu/pointMLP-pytorch/blob/main/classification_ScanObjectNN/ScanObjectNN.py
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

# os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

'''
def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'h5_files')):
        # note that this link only contains the hardest perturbed variant (PB_T50_RS).
        # for full versions, consider the following link.
        www = 'https://web.northeastern.edu/smilelab/xuma/datasets/h5_files.zip'
        # www = 'http://103.24.77.34/scanobjectnn/h5_files.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))
'''

def load_scanobjectnn_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    # h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    h5_name = '/data/jykim/h5_files/main_split_nobg/' + partition + '_objectdataset.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


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
    R = np.dot(Rz1, np.dot(Ry,Rz2)).astype('float32')
    pointcloud = R.dot(pointcloud.T).T
    return pointcloud


def random_point_dropout(pc, max_dropout_ratio=0.875):
    dropout_ratio = np.random.random() * max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
    if len(drop_idx) > 0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ScanObjectNNDataset(Dataset):
    def __init__(self, num_points, partition='training',
                 random_rotate=False, random_jitter=False, random_translate=False, random_dropout=False):
        self.data, self.label = load_scanobjectnn_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.random_rotate = random_rotate
        self.random_jitter = random_jitter
        self.random_translate = random_translate
        self.random_dropout = random_dropout

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]

        # Pre-process ?
        pointcloud = pointcloud - pointcloud.mean(axis=0, keepdims=True)
        # pointcloud = pointcloud / np.sqrt((pointcloud**2).sum(axis=-1, keepdims=True)).max()
        
        # pointcloud = pointcloud / np.linalg.norm(pointcloud, axis=1).max()
        if self.random_rotate:
            pointcloud = rotate_pointcloud(pointcloud)
        if self.random_jitter:
            pointcloud = jitter_pointcloud(pointcloud)
        if self.random_translate:
            pointcloud = translate_pointcloud(pointcloud)
        if self.random_dropout:
            pointcloud = random_point_dropout(pointcloud)
       
        if self.partition == 'training':
            np.random.shuffle(pointcloud)
        
        return torch.from_numpy(pointcloud), torch.from_numpy(np.array([label]).astype(np.int64))

    def __len__(self):
        return self.data.shape[0]


class ScanObjectNN:

    def __init__(self, num_points=2048, use_rotate=False, use_noisy=False, test=False, batch=16, workers=0):

        partition = 'test' if test else 'training'
        self.dataset = ScanObjectNNDataset(num_points = num_points,
                                           partition = partition,
                                           random_rotate = use_rotate,
                                           random_jitter = use_noisy,
                                           random_translate = False,
                                           random_dropout = False)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(0)
        '''
        self.dataloader = DataLoader(self.dataset,
                                     batch_size = batch,
                                     shuffle = True, # not test,
                                     num_workers = workers,
                                     worker_init_fn = lambda _ : np.random.seed())
        '''
        self.dataloader = DataLoader(self.dataset,
                                     batch_size = batch,
                                     shuffle = not test,
                                     num_workers = workers,
                                     worker_init_fn = seed_worker,
                                     generator = g)
        return


if __name__ == '__main__':

    from geo import *
    import matplotlib.pyplot as plt
    import pytorch3d.ops as ops

    '''
    def theta_bins(X):

        k = 16
        mc_num = 256

        X, F = sphere_coord_and_feature(X)
        B = X.size(0); N = X.size(1); D = F.size(-1)
        # old do sample logic
        Q, q_idx = ops.sample_farthest_points(X, K=mc_num, random_start_point=True)
        # KNN neighbor
        knn_result = ops.knn_points(Q, X, K=k, return_nn=True)
        X = knn_result.knn.view(B, -1, 3) # knn_X
        # knn_F = ops.knn_gather(F, knn_result.idx.clone()).view(B, -1, F.size(-1))

        B = X.size(0); N = X.size(1)
        X = X.view(B, N//k, k, 3)       # R^{B x #query x k x 3}
        Q = X[:, :, 0, :].unsqueeze(-2) # R^{B x #query x 1 x 3} which are the centroids of bundles
        Q = Q.expand(B, N//k, k, 3).reshape(B*N, 3)
        npole = torch.zeros_like(Q); npole[:, 2] += 1
        R_ref = rotation_between_s2points(Q, npole)
        X = torch.bmm(R_ref, X.view(-1, 3, 1))
        X = X.view(B, N, 3)
        theta = torch.acos(torch.clip(X[:, :, 2:3], min=-1.+1e-6, max=1.-1e-6))
        return theta.flatten()
 

    train = ScanObjectNN(1024, True, True, batch=1)
    test = ScanObjectNN(1024, test=True, batch=1)

    class_num = 11
    count = 0
    for i, (data, label) in enumerate(train.dataloader):
        if label.item() != class_num: continue
        count += 1
        thetas = theta_bins(data)
        thetas = thetas.numpy()
        hist, bins = np.histogram(thetas, bins=100)
        plt.clf()
        plt.xlim(0, 1.)
        plt.bar(bins[:-1], hist, width=np.diff(bins))
        plt.savefig(f'train_{count}.png')
        print(f'Train {count}')
        if count >= 10: break

    count = 0
    for i, (data, label) in enumerate(test.dataloader):
        if label.item() != class_num: continue
        count += 1
        thetas = theta_bins(data)
        thetas = theta_bins(data)
        thetas = thetas.numpy()
        hist, bins = np.histogram(thetas, bins=100)
        plt.clf()
        plt.xlim(0, 1.)
        plt.bar(bins[:-1], hist, width=np.diff(bins))
        plt.savefig(f'test_{count}.png')
 
        print(f'Test {count}')
        if count >= 10: break
    '''

    train = ScanObjectNN(1024, True, True, test=False, batch=16)
    test = ScanObjectNN(1024, test=True, batch=1)

    for i, (pc, label) in enumerate(train.dataloader):

        print(pc.shape)
        print(label.shape)

    '''
        pc = pc.squeeze(0)
        print(f'{i+1} PC')
        print(pc.mean(dim=0))
        print(pc.norm(dim=-1).max())
        pc = pc / pc.norm(dim=-1).max()
        print(pc.mean(dim=0))
        print(pc.shape)
        print(label.shape)
        if i >= 9: break

    print('########################################')
    for i, (pc, label) in enumerate(test.dataloader):

        pc = pc.squeeze(0)
        print(f'{i+1} PC')
        print(pc.mean(dim=0))
        print(pc.norm(dim=-1).max())
        pc = pc / pc.norm(dim=-1).max()
        print(pc.mean(dim=0))
        if i >= 9: break
    '''





