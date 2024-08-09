import torch
import torch.nn as nn
import numpy as np
import random

from sklearn.manifold import TSNE
from util.modelnet40 import ModelNet
from networks.models import SO3ModelNetMetric as Model

from tqdm import tqdm
from omegaconf import OmegaConf

import matplotlib
import matplotlib.pyplot as plt
import os

SEED = 1557
np.random.seed(SEED)
random.seed(SEED)
COLORS = [np.random.rand(3,) for _ in range(40)]
CMAP = matplotlib.colors.ListedColormap(COLORS)

def draw_tsne():

    print('1. Load model and compute feature vectors')
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
    modelnet = ModelNet(num_points=num_points, test=True, use_rotate=True, use_noisy=False, batch=args.batchsize, workers=4)
    dataloader = modelnet.dataloader
    label_list = modelnet.dataset.label
    pre_db = []
    met_db = []
    with torch.no_grad():
        for i, item in enumerate(tqdm(dataloader, desc='query vec reading', position=0)):
            pc, _ = item
            f_pre, f_met = model.get_feature(pc.to(device))
            pre_db.append(f_pre.to('cpu'))
            met_db.append(f_met.to('cpu'))
        pre_db = torch.cat(pre_db, dim=0).numpy().astype(np.float64)
        met_db = torch.cat(met_db, dim=0).numpy().astype(np.float64)
    print('2. Compute t-SNE')
    tsne = TSNE(random_state=0, n_iter=500)
    tsne_pre = tsne.fit_transform(pre_db)
    print('t-SNE of pretrained feature done')
    tsne = TSNE(random_state=0, n_iter=500)
    tsne_met = tsne.fit_transform(met_db)
    print('t-SNE of metrizable feature done')
    # np.savez('tsne_result.npz', pre=tsne_pre, met=tsne_met, label=label_list)
    print('3. t-SNE result is saved !')

    # plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    axes[0].scatter(tsne_pre[:, 0], tsne_pre[:, 1], c=label_list, cmap=CMAP, alpha=0.5, marker='.')
    axes[1].scatter(tsne_met[:, 0], tsne_met[:, 1], c=label_list, cmap=CMAP, alpha=0.5, marker='.')
    axes[0].set_title('Pre-trained feature', fontsize=24)
    axes[1].set_title('Metric feature', fontsize=24)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.tight_layout()
    plt.savefig('metric_tsne.png', bbox_inches='tight')


if __name__ == '__main__':

    draw_tsne()











