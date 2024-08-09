import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim

from util.loss import triplet, quadruplet, distance
from util.modelnet40_metric import ModelNet
from networks.models import SO3ModelNetMetric as Model

import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import matplotlib
import matplotlib.pyplot as plt

def main():

    # Load Checkpoint
    ckpt = torch.load('weight/modelnet/modelnet_metric_best.pth')
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
    except KeyError:
        num_points = 2048
        print('The number of points is set to default: 2048')
    modelnet = ModelNet(num_points=num_points, pos_num=1, neg_num=1, test=True, use_rotate=True, use_noisy=False, batch=args.batchsize, workers=4)
    dataloader = modelnet.dataloader
    label_list = modelnet.dataset.label
    query_db = []
    K = 240
    cls_name_dict = {}
    with torch.no_grad():
        for i, item in enumerate(tqdm(dataloader, desc='query vec reading', position=0)):
            anchor, _, _ = item
            q_vec = model(anchor.to(device))
            query_db.append(q_vec.to('cpu'))
        query_db = torch.cat(query_db, dim=0)
        N = query_db.shape[0]
        print(f'Query Database Shape: {query_db.shape}')
        query_diff = query_db.unsqueeze(1) - query_db.unsqueeze(0)
        query_diff = query_diff.norm(p=2, dim=-1)
        query_diff.fill_diagonal_(torch.tensor(float('inf')))
        dist, indices = torch.topk(query_diff, k=K, dim=-1, largest=False, sorted=True)
        print(f'Query Index {indices.shape}')

        count = 0
        precision_every = np.zeros(K)
        recall_every = np.zeros(K)
        recall_per_class = np.zeros((40, K))
        class_count = np.zeros((40, 1))
        precision_list = []
        for i, idx in enumerate(indices):
            ref_class = label_list[i]
            cls_name = modelnet.dataset.name[i]
            cls_name_dict[ref_class.item()] = cls_name
            retrieved_class = label_list[idx.tolist()]
            recall = (retrieved_class==ref_class)
            class_count[ref_class] += 1
            if recall.sum() == 0:
                precision_list.append(0.)
                continue
            recall = recall.flatten()
            
            precisionK = np.cumsum(recall) / (np.arange(K) + 1)
            precision_every += precisionK
            num_relevant = np.sum(label_list == ref_class)
            # ap = (precisionK * recall).sum() / num_relevant
            ap = (precisionK * recall).sum() / recall.sum()
            precision_list.append(ap)

            recall = np.cumsum(recall)
            recall = (recall > 0)
            recall_every += recall
            recall_per_class[ref_class, :] += recall
            count += 1
        one_percent = int(N / 100)
        print(f'1% = {one_percent}')
        print(f'Average Precision @1%')
        print(precision_every[:one_percent] / count)
        print(f'Average Recall @1%')
        print(recall_every[:one_percent] / count)
        print('mAP')
        print(np.mean(precision_list))
        class_recall_at_1 = (recall_per_class / class_count)[:, :1].flatten()
        print('Recall per class @1')
        print(class_recall_at_1)
        # Draw a horizontal bar plot of AR@1 per class
        cls_idx = np.arange(40)
        ar1_sorted_idx = class_recall_at_1.argsort()
        ar1 = class_recall_at_1[ar1_sorted_idx]
        cls_idx = cls_idx[ar1_sorted_idx]
        cls_bar = [cls_name_dict[cls_idx[i]] for i in range(40)]
        fig = plt.figure(figsize=(18, 14))
        ax = fig.add_subplot()
        ax.barh(np.arange(40), ar1, height=0.6, align='center', color=plt.get_cmap('bwr_r')(ar1), edgecolor='black')
        ax.set_yticks(np.arange(40), labels=cls_bar, fontsize=24)
        ax.tick_params(axis='x', labelsize=24)
        ax.invert_yaxis()
        ax.set_xlabel('Average Recall @1', fontsize=30)
        ax.margins(y=0.01, x=0.02)
        plt.tight_layout()
        plt.savefig('ar1_barh.png')
        return
 

if __name__ == '__main__':

    main()





