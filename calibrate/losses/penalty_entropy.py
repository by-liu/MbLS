import torch
import torch.nn as nn
import torch.nn.functional as F

from calibrate.utils.constants import EPS


class PenaltyEntropy(nn.Module):
    """Regularizing neural networks by penalizing confident output distributions, 2017. <https://arxiv.org/pdf/1701.06548>

        loss = CE - alpha * Entropy(p)
    """
    def __init__(self, alpha=1.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

    @property
    def names(self):
        return "loss", "loss_ce", "loss_ent"

    def forward(self, inputs, targets):
        if inputs.dim() > 2:
            inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
            inputs = inputs.transpose(1, 2)    # N,C,H*W => N,H*W,C
            inputs = inputs.contiguous().view(-1, inputs.size(2))   # N,H*W,C => N*H*W,C
            targets = targets.view(-1)

        if self.ignore_index >= 0:
            index = torch.nonzero(targets != self.ignore_index).squeeze()
            inputs = inputs[index, :]
            targets = targets[index]

        # cross entropy
        loss_ce = F.cross_entropy(inputs, targets)

        # entropy
        prob = F.log_softmax(inputs, dim=1).exp()
        prob = torch.clamp(prob, EPS, 1.0 - EPS)
        ent = - prob * torch.log(prob)
        loss_ent = ent.mean()

        loss = loss_ce - self.alpha * loss_ent

        return loss, loss_ce, loss_ent
