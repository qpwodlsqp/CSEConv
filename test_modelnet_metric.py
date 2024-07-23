import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim

from util.loss import triplet, quadruplet, distance
from util.modelnet40_metric import ModelNet
from networks.models import SO3ModelNetMetric as Model

import argparse
import time
import wandb
import datetime
import os
from tqdm import tqdm
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='ModelNet40 Classification',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--cfg', type=str, default=None, metavar='N',
                    help='configuration file name for the model architecture')

parser.add_argument('--use_rotate', action='store_true', default=False,
                    help='whether to augment train data with random SO3 rotation')
parser.add_argument('--use_noisy', action='store_true', default=False,
                    help='whether to augment train data other noisness')
parser.add_argument('--batchsize', type=int, default=16, metavar='N',
                    help='mini-batch size')
parser.add_argument('--seed', type=int, default=1557, metavar='N',
                    help='random seed')
parser.add_argument('--log_every', type=int, default=300, metavar='N',
                    help='logging interval')
args = parser.parse_args()

def main():

    # Load Checkpoint
    ckpt = torch.load('weight/modelnet/metric_best_pretrained.pth')
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
    now = datetime.datetime.now()
    #torch.autograd.set_detect_anomaly(True)
    query_db = []
    K = 240
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
        precision_list = []
        for i, idx in enumerate(indices):
            ref_class = label_list[i]
            retrieved_class = label_list[idx.tolist()]
            recall = (retrieved_class==ref_class)
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
            count += 1
        one_percent = int(N / 100)
        print(f'1% = {one_percent}')
        print(f'Average Precision @1%')
        print(precision_every[:one_percent] / count)
        print(f'Average Recall @1%')
        print(recall_every[:one_percent] / count)
        print('mAP')
        print(np.mean(precision_list))
 
if __name__ == '__main__':

    if not os.path.exists('result'):
        os.makedirs('result')
    main()





