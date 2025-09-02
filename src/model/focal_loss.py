import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):

        if len(targets.shape) > 1 and targets.shape[1] > 1:
            targets = torch.argmax(targets, dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        logp = F.log_softmax(inputs, dim=1)
        prob = torch.exp(logp)
        
        pt = prob.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss