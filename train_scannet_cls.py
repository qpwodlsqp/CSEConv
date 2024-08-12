import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim

from util.ScanObjectNN import ScanObjectNN
from networks.models import SO3ScanNN as Model

import argparse
import wandb
import datetime
import os
import shutil

from tqdm import tqdm
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description='ScanObjectNN Classification',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--cfg', type=str, default=None, metavar='N',
                    help='configuration file name for the model architecture')
parser.add_argument('--wandb_id', type=str, default=None, metavar='N',
                    help='your wandb id')

parser.add_argument('--use_scheduler', action='store_true', default=False,
                    help='whether to cosine annealing scheduler for learning rate')
parser.add_argument('--use_rotate', action='store_true', default=False,
                    help='whether to augment train data with random SO3 rotation')
parser.add_argument('--use_noisy', action='store_true', default=False,
                    help='whether to augment train data other noisness')
parser.add_argument('--batchsize', type=int, default=8, metavar='N',
                    help='mini-batch size')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=1e-4, metavar='N',
                    help='initial learning rate')
parser.add_argument('--filter_decay', type=float, default=1e-3, metavar='N',
                    help='weight decay rate for filter function')
parser.add_argument('--seed', type=int, default=1601, metavar='N',
                    help='random seed')
parser.add_argument('--log_every', type=int, default=289, metavar='N',
                    help='logging interval')
args = parser.parse_args()

assert args.cfg is not None
assert args.wandb_id is not None
cfg_path = os.path.join(os.getcwd(), 'cfg', 'scanobjectnn', f'{args.cfg}.yaml')
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
    model = Model(cfg_dict, class_dim=15)
    model.to(device)
    model.train()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    modelnet = ScanObjectNN(num_points=num_points, test=False,
                            use_rotate=args.use_rotate,
                            use_noisy=args.use_noisy,
                            batch=args.batchsize, workers=4)
    modelnet_valid = ScanObjectNN(num_points=num_points, test=True,
                                  use_rotate=True,
                                  use_noisy=args.use_noisy, 
                                  batch=args.batchsize, workers=4)
    trainloader = modelnet.dataloader
    validloader = modelnet_valid.dataloader
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)

    now = datetime.datetime.now()
    now_str = now.strftime('%m-%d-%H-%M-%S')
    nick = f"CSEConv_{args.cfg}_{now_str}"
    os.makedirs(f"result/{nick}")
    wandb.init(project='ScanObjectNN Classification', entity=args.wandb_id, config=vars(args))
    wandb.run.name = f'{nick}'

    #torch.autograd.set_detect_anomaly(True)
    class_dim = 15
    run_loss = []
    i_step = 0
    best_acc = 0.
    best_step = 0
    for epoch in tqdm(range(args.epochs), desc='total epoch', position=0):

        for i, item in enumerate(tqdm(trainloader, desc='current epoch', position=1, leave=False)):
            i_step += 1
            pc, label = item
            # When using naive Adam
            optimizer.zero_grad()
            pc, label = item
            pred = model(pc.to(device))
            loss = criterion(pred, label.flatten().to(device))

            run_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            if i_step % args.log_every == 0:

                with torch.no_grad():
                    model.eval()
                    ans = torch.zeros(class_dim, 2)
                    for i, item in enumerate(tqdm(validloader, desc='test loader', position=2, leave=False)):
                        pc, label = item
                        pred = model(pc.to(device)).argmax(dim=-1).to('cpu')
                        label = label.squeeze(1)
                        ans[:, 0] += label.bincount(minlength=class_dim)
                        correct = pred[torch.where(pred == label)[0]]
                        ans[:, 1] += correct.bincount(minlength=class_dim)
                    acc_per_class = ans[:, 1] / ans[:, 0]
                    test_total_acc = ans[:, 1].sum() / ans[:, 0].sum()

                    ans = torch.zeros(class_dim, 2)
                    for i, item in enumerate(tqdm(trainloader, desc='train loader', position=2, leave=False)):
                        pc, label = item
                        pred = model(pc.to(device)).argmax(dim=-1).to('cpu')
                        label = label.squeeze(1)
                        ans[:, 0] += label.bincount(minlength=class_dim)
                        correct = pred[torch.where(pred == label)[0]]
                        ans[:, 1] += correct.bincount(minlength=class_dim)
                    model.train()
                    acc_per_class = ans[:, 1] / ans[:, 0]
                    train_total_acc = ans[:, 1].sum() / ans[:, 0].sum()

                wandb.log({'train loss': np.mean(run_loss),
                           'test_total_acc': np.mean(test_total_acc.item()),
                           'train_total_acc': np.mean(train_total_acc.item())}, i_step//args.log_every - 1)
                if best_acc < np.mean(test_total_acc.item()):
                    best_step = i_step
                    best_acc = np.mean(test_total_acc.item())
                    save_dict = {'args': args,
                                 'model_state_dict': model.state_dict(),
                                 'opt_state_dict': optimizer.state_dict()}
                    if args.use_scheduler:
                        save_dict['sched_state_dict'] = scheduler.state_dict()
                    torch.save(save_dict, os.path.join('result', nick, f'best.pth'))
                run_loss = []
        if args.use_scheduler:
            scheduler.step()

    if not os.path.exists(os.path.join('weight', 'scannet')):
        os.makedirs(os.path.join('weight', 'scannet'))
    weight_file_name = 'scannet_cls_rotated_best.pth' if args.use_rotate else 'scannet_cls_best.pth'
    shutil.copy(os.path.join('result', nick, f"best.pth"), os.path.join('weight', 'scannet', weight_file_name))
    print(f'Best Step: {best_step}')
    print(f'Best Accuracy: {best_acc}')


if __name__ == '__main__':

    if not os.path.exists('result'):
        os.makedirs('result')
    main()





