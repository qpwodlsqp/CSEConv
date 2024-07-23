import torch
import torch.nn.functional as F

def triplet(anchor, pos, neg, pos_num, neg_num, alpha=0.1, metric='L2', lazy=False):
    B = anchor.size(0)
    pos = pos.view(B, pos_num, -1)
    neg = neg.view(B, neg_num, -1)
    pos_dist = distance(anchor.unsqueeze(1).repeat(1, pos_num, 1), pos, metric).square() # B x pos_num
    pos_dist = pos_dist.max(dim=-1, keepdim=True).values # B x 1
    pos_dist = pos_dist.repeat(1, neg_num)               # B x neg_num
    neg_dist = distance(anchor.unsqueeze(1).repeat(1, neg_num, 1), neg, metric).square() # B x neg_num
    d = F.relu(pos_dist - neg_dist + alpha)
    if lazy:
        d = d.max(dim=-1).values
    return torch.mean(d)

def quadruplet(anchor, pos, neg1, neg2, pos_num, neg_num, alpha=0.1, beta=0.1, metric='L2', lazy=True):
    B = anchor.size(0)
    pos  = pos.view(B, pos_num, -1)
    neg1 = neg1.view(B, neg_num, -1)
    pos_dist  = distance(anchor.unsqueeze(1).repeat(1, pos_num, 1), pos, metric).square() # B x pos_num
    pos_dist  = pos_dist.max(dim=-1, keepdim=True).values # B x 1
    pos_dist  = pos_dist.repeat(1, neg_num)               # B x neg_num
    neg1_dist = distance(anchor.unsqueeze(1).repeat(1, neg_num, 1), neg1, metric).square() # B x neg_num
    neg2_dist = distance(neg1, neg2.unsqueeze(1).repeat(1, neg_num, 1), metric).square()  # B x neg_num
    d1 = F.relu(pos_dist - neg1_dist + alpha)
    d2 = F.relu(pos_dist - neg2_dist + beta)
    if lazy:
        d1 = d1.max(dim=-1).values
        d2 = d2.max(dim=-1).values
    return torch.mean(d1 + d2) # , d1.mean(), d2.mean()

def distance(x, y, metric):

    if metric == 'L2':
        return torch.norm(x - y, p=2, dim=-1)
    elif metric == 'L1':
        return torch.norm(x - y, p=1, dim=-1)
    elif metric == 'cos':
        # return torch.acos(torch.clamp((x * y).sum(dim=-1), -1. + 1e-6, 1. - 1e-6))
        return 0.5 * torch.norm(x - y, p=2, dim=-1).square() # equivalent to 1 - \cos(x, y) if |x| = |y| = 1
