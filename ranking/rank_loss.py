import torch
import torch.nn as nn
import torch.nn.functional as F

class ListLoss(nn.Module):
    def __init__(self):
        super(ListLoss, self).__init__()

    def forward(self, scores, true_ranks):
        return self.rank_loss(scores, true_ranks)

    def rank_loss(self, scores, true_ranks):
        prob_pred = F.softmax(scores, dim=-1)
        true_prob = F.softmax(true_ranks.float(), dim=-1)
        loss = F.kl_div(prob_pred.log(), true_prob, reduction='batchmean')
        
        return loss
