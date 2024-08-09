import torch
import torch.nn as nn
import numpy as np
import random

from util.modelnet40 import ModelNet
from networks.models import SO3ModelNet as Model

import argparse
import time
import wandb
import datetime
import os
from tqdm import tqdm
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='ScanObjectNN Classification Test',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--use_rotate', action='store_true', default=False,
                    help='whether to test the model trained with SO3 augmentation')
args_test = parser.parse_args()

def main():

    CLS_CHOICE = ['airplane', 'bathtub', 'bed', 'bench','bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', \
                  'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', \
                  'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', \
                  'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    CLS_CHOICE = np.array(CLS_CHOICE)

    # Load Checkpoint
    if args_test.use_rotate:
        model_name = 'modelnet_cls_rotated_best.pth'
    else:
        model_name = 'modelnet_cls_best.pth'
    ckpt = torch.load(f'weight/modelnet/{model_name}')
    args = ckpt['args'] # configurations 
    # Reproducibility
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # Load Model
    device = torch.device("cuda")
    cfg_dict = OmegaConf.load(f'cfg/modelnet/{args.cfg}.yaml')
    model = Model(cfg_dict)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    # Load Dataset
    try:
        num_points = cfg_dict['num_points']
        print(f'The number of points is set to {num_points}')
    except KeyError:
        num_points = 2048
        print('The number of points is set to default: 2048')
    class_dim = 40
    modelnet = ModelNet(num_points=num_points, use_rotate=False, use_noisy=False, test=True, batch=args.batchsize, workers=4)
    testloader = modelnet.dataloader
    # Classification Test
    ans = torch.zeros(class_dim, 2)
    with torch.no_grad():
        for i, item in enumerate(tqdm(testloader, desc='test loader', position=1, leave=True)):
            pc, label = item
            pred = model(pc.to(device)).argmax(dim=-1).to('cpu')
            label = label.squeeze(1)
            ans[:, 0] += label.bincount(minlength=class_dim)
            correct = pred[torch.where(pred == label)[0]]
            ans[:, 1] += correct.bincount(minlength=class_dim)
    acc_per_class = ans[:, 1] / ans[:, 0]
    total_acc = ans[:, 1].sum() / ans[:, 0].sum()
    print(f'\n[Clean] [Id] Total Accuracy: {total_acc}')
    print('Accuracy per Class')
    print(acc_per_class.sort(descending=True).values)
    arg = torch.argsort(acc_per_class, descending=True)
    print(CLS_CHOICE[arg.numpy()])

    del modelnet
    modelnet = ModelNet(num_points=num_points, use_rotate=True, use_noisy=False, test=True, batch=args.batchsize, workers=4)
    testloader = modelnet.dataloader
    # Classification Test
    ans = torch.zeros(class_dim, 2)
    with torch.no_grad():
        for i, item in enumerate(tqdm(testloader, desc='test loader', position=1, leave=True)):
            pc, label = item
            pred = model(pc.to(device)).argmax(dim=-1).to('cpu')
            label = label.squeeze(1)
            ans[:, 0] += label.bincount(minlength=class_dim)
            correct = pred[torch.where(pred == label)[0]]
            ans[:, 1] += correct.bincount(minlength=class_dim)
    acc_per_class = ans[:, 1] / ans[:, 0]
    total_acc = ans[:, 1].sum() / ans[:, 0].sum()
    print(f'\n[Clean] [SO3] Total Accuracy: {total_acc}')
    print('Accuracy per Class')
    print(acc_per_class.sort(descending=True).values)
    arg = torch.argsort(acc_per_class, descending=True)
    print(CLS_CHOICE[arg.numpy()])

if __name__ == '__main__':

    main()





