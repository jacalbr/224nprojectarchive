# Reference from https://github.com/archinetai/vat-pytorch/blob/main/vat_pytorch/utils.py

import torch
import torch.nn.functional as F
#======================online contrastive======================
def online_contrastive_loss(logits, labels, margin= 0.5):
    distance = 1 - logits
    negatives = distance[labels == 0]
    positives = distance[labels == 1]

    # select hard positive and hard negative pairs
    negative_pairs = negatives[negatives < (positives.max() if len(positives) > 1 else negatives.mean())]
    positive_pairs = positives[positives > (negatives.min() if len(negatives) > 1 else positives.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    loss = positive_loss + negative_loss
    return loss



#======================Smart======================
def inf_norm(x):
    return torch.norm(x, p=float("inf"), dim=-1, keepdim=True)


def kl_loss(input, target, reduction="batchmean"):
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )


def sym_kl_loss(input, target, reduction="batchmean", alpha=1.0):
    return alpha * F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target.detach(), dim=-1),
        reduction=reduction,
    ) + F.kl_div(
        F.log_softmax(target, dim=-1),
        F.softmax(input.detach(), dim=-1),
        reduction=reduction,
    )
