import torch
import torch.nn as nn
import numpy as np
import random

from util.modelnet40 import ModelNet
from networks.models import SO3ModelNet as Model
from sklearn.metrics import accuracy_score

import argparse
import time
import wandb
import datetime
import os
from tqdm import tqdm
from omegaconf import OmegaConf


def main():

    CLS_CHOICE = ['airplane', 'bathtub', 'bed', 'bench','bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone','cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
    CLS_CHOICE = np.array(CLS_CHOICE)

    # Load Checkpoint
    ckpt = torch.load('weight/modelnet/best.pth')
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
    # iso_loss(model)
    # Load Dataset
    try:
        num_points = cfg_dict['num_points']
        print(f'The number of points is set to {num_points}')
    except KeyError:
        num_points = 2048
        print('The number of points is set to default: 2048')
    class_dim = 40
    highest_mem = 0.
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
        mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024)
        torch.cuda.reset_peak_memory_stats()
        if mem_used_max_GB > highest_mem:
            highest_mem = mem_used_max_GB
    acc_per_class = ans[:, 1] / ans[:, 0]
    total_acc = ans[:, 1].sum() / ans[:, 0].sum()
    print(f'\n[Clean] [Id] Total Accuracy: {total_acc}')
    print('Accuracy per Class')
    print(f'MAX MEM: {highest_mem}')
    print(acc_per_class.sort(descending=True).values)
    arg = torch.argsort(acc_per_class, descending=True)
    print(CLS_CHOICE[arg.numpy()])

    '''
    ans = torch.zeros(class_dim, 2)
    pred_stack = []
    label_stack = []
    total_acc = 0.
    with torch.no_grad():
        for i, item in enumerate(tqdm(testloader, desc='test loader sklearn', position=1, leave=True)):
            pc, label = item
            pred = model(pc.to(device)).argmax(dim=-1).to('cpu')
            label = label.squeeze(1)
            # print(pred)
            # print(label)
            # print('################')
            pred_stack.append(pred)
            label_stack.append(label)
        pred_stack = torch.cat(pred_stack, dim=0).numpy()
        label_stack = torch.cat(label_stack, dim=0).numpy()
        total_acc = accuracy_score(label_stack, pred_stack)

    print(f'\n[Clean] [Id] Total Accuracy: {total_acc}')
    print('Accuracy per Class')
    print(f'MAX MEM: {highest_mem}')
    print(acc_per_class)
    '''

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

    if not os.path.exists('result'):
        os.makedirs('result')
    main()





