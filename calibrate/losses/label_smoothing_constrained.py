import torch
import torch.nn as nn
import torch.nn.functional as F
import math


_EPS = 1e-10


class LabelSmoothConstrainedLoss(nn.Module):
    """ Combine CE with a log-barrier constraints on logits:
    Formulation of log-barrier:
        loss = - 1 / t * log(-z) {z <= - 1 / t^2};
            tz - 1 / t * log(1 / t^2) + 1 / t {z > - 1 / t^2}
            (z = abs_diff - margin)
    """
    def __init__(self, margin=10.0, t_logb=10.0, w_logb=0.1,
                 mu=0, schedule="", max_t=100.0, step_size=100):
        super().__init__()
        assert schedule in ("", "add", "multiply", "step")
        self.margin = margin
        self.t_logb = t_logb
        self.w_logb = w_logb
        self.mu = mu
        self.schedule = schedule
        self.max_t = max_t
        self.step_size = step_size

    @property
    def names(self):
        return "loss", "loss_ce", "loss_barrier"

    def schedule_t(self, epoch):
        if self.schedule == "add":
            self.t_logb = min(self.t_logb + self.mu, self.max_t)
        elif self.schedule == "multiply":
            self.t_logb = min(self.t_logb * self.mu, self.max_t)
        elif self.schedule == "step":
            if (epoch + 1) % self.step_size == 0:
                self.t_logb = min(self.t_logb * self.mu, self.max_t)

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        return max_values - inputs

    def forward(self, inputs, targets):
        # cross entropy
        logprobs = F.log_softmax(inputs, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=targets.unsqueeze(1))
        data_loss = nll_loss.squeeze(1)

        # log barrier
        abs_diffs = self.get_diff(inputs)
        z = abs_diffs - self.margin

        # The piece-wise function
        log_barrier_loss = torch.where(
            z < - 1 / (self.t_logb ** 2),
            - (1 / self.t_logb) * torch.log(- z + _EPS),
            self.t_logb * z - (1 / self.t_logb) * math.log(1 / (self.t_logb ** 2)) + 1 / self.t_logb
        )

        data_loss = data_loss.mean()
        log_barrier_loss = log_barrier_loss.mean()
        loss = data_loss + self.w_logb * log_barrier_loss

        return loss, data_loss, log_barrier_loss

    # def forward(self, input, target):
    #     logprobs = F.log_softmax(input, dim=-1)
    #     nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
    #     nll_loss = nll_loss.squeeze(1)

    #     maxValues = input.max(dim=1)
    #     maxValues = maxValues.values

    #     absDiffs = torch.empty((input.shape))

    #     for i in range(input.shape[1]):
    #         absDiffs[:,i] = maxValues-input[:,i]

    #     # Log barrier (constraint on distances)
    #     t_logbTensor = torch.FloatTensor(1)
    #     t_logbTensor.fill_(self.t_logb)

    #     z_val = absDiffs - self.margin
    #     z_val_first_term = (z_val < -(1 / (self.t_logb ** 2))).type(torch.FloatTensor)
    #     z_val_second_term = 1 - z_val_first_term

    #     if torch.cuda.is_available():
    #         z_val_first_term = z_val_first_term.cuda()
    #         z_val_second_term = z_val_second_term.cuda()
    #         t_logbTensor = t_logbTensor.cuda()
    #         z_val=z_val.cuda()

    #     lossA = z_val_first_term * (-(1 / t_logbTensor) * torch.log(-z_val))
    #     # Trick to remove NaNs
    #     lossA[lossA != lossA] = 0
    #     lossB = z_val_second_term * (t_logbTensor * z_val - (1 / t_logbTensor) * torch.log(1 / (t_logbTensor ** 2)) + 1 / t_logbTensor)

    #     log_barrier_loss = lossA + lossB

    #     data_loss = nll_loss

    #     #if (log_barrier_loss.mean()=='nan'):
    #     #    pdb.set_trace()

    #     #return data_loss.mean()+0.1*log_barrier_loss.mean()

    #    # return data_loss.sum()+0.5*log_barrier_loss.sum()
    #     return [data_loss.sum(), log_barrier_loss.sum()]