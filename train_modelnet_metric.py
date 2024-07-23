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
parser.add_argument('--alpha', type=float, default=0.5, metavar='N',
                    help='hyperparam for triplet & quadruplet')
parser.add_argument('--beta', type=float, default=0.2, metavar='N',
                    help='hyperparam for quadruplet')
parser.add_argument('--use_cos', action='store_true', default=False,
                    help='whether to use cosine distance')
parser.add_argument('--use_quad', action='store_true', default=False,
                    help='whether to use quadruplet loss')
parser.add_argument('--pos_num', type=int, default=2, metavar='N',
                    help='mini-batch size')
parser.add_argument('--neg_num', type=int, default=10, metavar='N',
                    help='mini-batch size')

parser.add_argument('--use_rotate', action='store_true', default=False,
                    help='whether to augment train data with random SO3 rotation')
parser.add_argument('--use_noisy', action='store_true', default=False,
                    help='whether to augment train data other noisness')
parser.add_argument('--batchsize', type=int, default=16, metavar='N',
                    help='mini-batch size')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='initial learning rate')
parser.add_argument('--seed', type=int, default=1557, metavar='N',
                    help='random seed')
parser.add_argument('--log_every', type=int, default=300, metavar='N',
                    help='logging interval')
parser.add_argument('--use_scheduler', action='store_true', default=False,
                    help='whether to cosine annealing scheduler for learning rate')
args = parser.parse_args()

assert args.cfg is not None
cfg_path = os.path.join(os.getcwd(), 'cfg', 'modelnet', f'{args.cfg}.yaml')
assert os.path.isfile(cfg_path)

def main():

    torch.backends.cudnn.benchmark = True
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda")
    cfg_dict = OmegaConf.load(cfg_path)
    try:
        num_points = cfg_dict['num_points']
    except KeyError:
        num_points = 2048
        print('The number of points is set to default: 2048')
    model = Model(cfg_dict)
    print('Load Classifier Model')
    ckpt = torch.load('weight/modelnet/pretrained.pth')
    pretrained = ckpt['model_state_dict'] 
    conv_state_dict = {}
    bn_state_dict = {}
    enc_state_dict = {}
    for name, weight in pretrained.items():
        if 'conv_layers.' in name:
            conv_state_dict[name[12:]] = weight
    for name, weight in pretrained.items():
        if 'bn_layers.' in name:
            bn_state_dict[name[10:]] = weight
    for name, weight in pretrained.items():
        if 'pointnet.' in name:
            enc_state_dict[name[9:]] = weight

    model.conv_layers.load_state_dict(conv_state_dict)
    model.bn_layers.load_state_dict(bn_state_dict)
    model.pointnet.load_state_dict(enc_state_dict)

    print('freeze backbone')
    for param in model.conv_layers.parameters():
        param.requires_grad = False
    for param in model.bn_layers.parameters():
        param.requires_grad = False
    for param in model.pointnet.parameters():
        param.requires_grad = False
    # print('no freeze')
    model.to(device)
    model.train()
    # modelnet = ModelNet(num_points=num_points, test=False, use_rotate=args.use_rotate, use_noisy=args.use_noisy, batch=args.batchsize, workers=4)
    modelnet = ModelNet(num_points=num_points, test=False, pos_num=args.pos_num, neg_num=args.neg_num, use_rotate=args.use_rotate, use_noisy=args.use_noisy, use_quad=args.use_quad, batch=args.batchsize, workers=4)
    modelnet_valid = ModelNet(num_points=num_points, pos_num=1, neg_num=1, use_rotate=True, use_noisy=args.use_noisy, use_quad=args.use_quad, test=True, batch=args.batchsize, workers=4)
    trainloader = modelnet.dataloader
    validloader = modelnet_valid.dataloader
    label_list  = modelnet_valid.dataset.label
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    now = datetime.datetime.now()
    now_str = now.strftime('%m-%d-%H-%M-%S')
    nick = f"NEW_SO3-Metric-Pretrain_{args.cfg}_{now_str}"
    os.makedirs(f"result/{nick}")
    wandb.init(project='ModelNet40 Metric Learning', entity='qpwodlsqp', config=vars(args))
    wandb.run.name = f'{nick}'

    #torch.autograd.set_detect_anomaly(True)
    class_dim = 40
    run_loss = []
    krr_reg_list = []
    basis_omega_list = []
    train_pos_dist = []
    train_neg_dist = []

    metric = 'cos' if args.use_cos else 'L2'
    i_step = 0
    best_mAP = 0.
    best_step = 0
    for epoch in tqdm(range(args.epochs), desc='total epoch', position=0):

        for i, item in enumerate(tqdm(trainloader, desc='current epoch', position=1, leave=False)):
            i_step += 1
            optimizer.zero_grad()
            if args.use_quad:
                anchor, pos, neg, neg2 = item
            else:
                anchor, pos, neg = item
            q_vec   = model(anchor.to(device))
            pos = pos.flatten(0, 1).to(device)
            neg = neg.flatten(0, 1).to(device)
            pos_vec = model(pos)
            neg_vec = model(neg)
            loss = triplet(q_vec, pos_vec, neg_vec, args.pos_num, args.neg_num, args.alpha, metric)
            
            if not args.use_quad:
                loss = triplet(q_vec, pos_vec, neg_vec, args.pos_num, args.neg_num, args.alpha, metric)
            else:
                neg2_vec = model(neg2.to(device))
                loss = quadruplet(q_vec, pos_vec, neg_vec, neg2_vec, args.pos_num, args.neg_num, args.alpha, args.beta, metric)
            
            with torch.no_grad():
                B = q_vec.size(0)
                train_pos_dist.append(distance(q_vec.unsqueeze(1), pos_vec.view(B, args.pos_num, -1), metric).mean().item())
                train_neg_dist.append(distance(q_vec.unsqueeze(1), neg_vec.view(B, args.neg_num, -1), metric).mean().item())
            run_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            if i_step % args.log_every == 0:
                valid_loss = []
                valid_pos_dist = []
                valid_neg_dist = []
                with torch.no_grad():
                    query_db = []
                    K = 240
                    for i, item in enumerate(tqdm(validloader, desc='query vec reading', position=0)):
                        if args.use_quad:
                            anchor, _, _, _ = item
                        else:
                            anchor, _, _ = item
                        q_vec = model(anchor.to(device))
                        query_db.append(q_vec.to('cpu'))
                    query_db = torch.cat(query_db, dim=0)
                    query_diff = query_db.unsqueeze(1) - query_db.unsqueeze(0)
                    query_diff = query_diff.norm(p=2, dim=-1)
                    # dist, indices = torch.topk(query_diff, k=K+1, dim=-1, largest=False, sorted=True)
                    query_diff.fill_diagonal_(torch.tensor(float('inf')))
                    dist, indices = torch.topk(query_diff, k=K, dim=-1, largest=False, sorted=True)

                    count = 0
                    recall_every = np.zeros(K)
                    precision_list = []
                    for i, idx in enumerate(indices):
                        # idx = idx[1:]
                        ref_class = label_list[i]
                        retrieved_class = label_list[idx.tolist()]
                        recall = (retrieved_class==ref_class)
                        if recall.sum() == 0:
                            precision_list.append(0.)
                            continue
                        recall = recall.flatten()
                        
                        precisionK = np.cumsum(recall) / (np.arange(K) + 1)
                        num_relevant = np.sum(label_list == ref_class)
                        # ap = (precisionK * recall).sum() / num_relevant
                        ap = (precisionK * recall).sum() / recall.sum()
                        precision_list.append(ap)

                        recall = np.cumsum(recall)
                        recall = (recall > 0)
                        recall_every += recall
                        count += 1
                    mAP = np.mean(precision_list)
                    if mAP > best_mAP:
                        best_mAP = mAP
                        best_step = i_step
                        save_dict = {'args': args,
                                     'model_state_dict': model.state_dict(),
                                     'opt_state_dict': optimizer.state_dict()}
                        if args.use_scheduler:
                            save_dict['sched_state_dict'] = scheduler.state_dict()
                        torch.save(save_dict, os.path.join('result', nick, f'best.pth'))

                wandb.log({'train loss': np.mean(run_loss),
                           # 'valid loss': np.mean(valid_loss),
                           # 'KRR Reg'   : np.mean(krr_reg_list),
                           # 'Omega Reg' : np.mean(basis_omega_list),
                           'train pos dist': np.mean(train_pos_dist),
                           'train neg dist': np.mean(train_neg_dist),
                           # 'valid pos dist': np.mean(valid_pos_dist),
                           # 'valid neg dist': np.mean(valid_neg_dist),
                           'mAP': mAP}, i_step//args.log_every - 1)
                run_loss = []
                krr_reg_list = []
                basis_omega_list = []
                train_pos_dist = []
                train_neg_dist = []
        if args.use_scheduler:
            scheduler.step()
        '''
        save_dict = {'args': args,
                     'model_state_dict': model.state_dict(),
                     'opt_state_dict': optimizer.state_dict()}
        if args.use_scheduler:
            save_dict['sched_state_dict'] = scheduler.state_dict()
        torch.save(save_dict, os.path.join('result', nick, f'{epoch}.pth'))
        '''

if __name__ == '__main__':

    if not os.path.exists('result'):
        os.makedirs('result')
    main()





