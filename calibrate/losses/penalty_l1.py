import torch
import torch.nn as nn
import torch.nn.functional as F

from calibrate.utils.constants import EPS


class PenaltyL1(nn.Module):
    """Penalty L1
        loss = CE + alpha * |s - 1/K| (s: softmax outputs, K : number of classes)
    """
    def __init__(self, num_classes, alpha=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha

    @property
    def names(self):
        return "loss", "loss_ce", "loss_l1"

    def forward(self, inputs, targets):
        # cross_entropy
        loss_ce = F.cross_entropy(inputs, targets)

        # l1
        s = F.log_softmax(inputs, dim=1).exp()
        loss_l1 = (s - 1.0 / self.num_classes).abs()
        loss_l1 = loss_l1.sum() / inputs.shape[0]
        # loss_l1 = loss_l1.mean()

        loss = loss_ce + self.alpha * loss_l1

        return loss, loss_ce, loss_l1
