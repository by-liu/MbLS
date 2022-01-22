import time
import json
import numpy as np
import torch
from omegaconf import DictConfig
import logging
import wandb

from calibrate.engine.trainer import Trainer
from calibrate.losses import LogitMarginL1
from calibrate.utils import (
    round_dict
)
from calibrate.utils.torch_helper import to_numpy, get_lr

logger = logging.getLogger(__name__)


class TrainerMixup(Trainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def mixup_data(self, x, y):
        alpha = self.cfg.train.mixup.alpha
        lam = np.random.beta(alpha, alpha)

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        loss_a = self.loss_func(pred, y_a)
        loss_b = self.loss_func(pred, y_b)
        if isinstance(loss_a, tuple):
            loss = [lam * a + (1 - lam) * b for a, b in zip(loss_a, loss_b)]
        else:
            loss = lam * loss_a + (1 - lam) * loss_b

        return loss

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        self.train_total = 0
        self.train_correct = 0
        end = time.time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            mixed_x, y_a, y_b, lam = self.mixup_data(inputs, labels)
            # forward
            outputs = self.model(mixed_x)
            loss = self.mixup_criterion(outputs, y_a, y_b, lam)
            if isinstance(loss, (tuple, list)):
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            if self.cfg.train.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
            self.optimizer.step()
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            _, pred = torch.max(outputs, 1)
            self.train_total += pred.size(0)
            self.train_correct += (
                lam * (pred == y_a).detach().sum().float()
                + (1 - lam) * (pred == y_b).detach().sum().float()
            ).item()
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_epoch_info(epoch)

    def log_iter_info(self, iter, max_iter, epoch, phase="Train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict.update(self.loss_meter.get_vals())
        if phase.lower() != "train":
            log_dict.update(self.evaluator.curr_score())
            log_dict.update(self.logits_evaluator.curr_score())
        logger.info("{} Iter[{}/{}][{}]\t{}".format(
            phase, iter + 1, max_iter, epoch + 1,
            json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable and phase.lower() == "train":
            wandb_log_dict = {"iter": epoch * max_iter + iter}
            wandb_log_dict.update(dict(
                ("{}/Iter/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb.log(wandb_log_dict)

    def log_epoch_info(self, epoch, phase="Train"):
        log_dict = {}
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, LogitMarginL1):
            log_dict["alpha"] = self.loss_func.alpha

        if phase.lower() == "train":
            log_dict["samples"] = self.train_total
            log_dict["acc"] = self.train_correct / self.train_total
        else:
            log_dict["samples"] = self.evaluator.num_samples()
            metric, table_data = self.evaluator.mean_score(print=False)
            log_dict.update(metric)
            log_dict.update(self.logits_evaluator.mean_score())
        # log_dict.update(self.probs_evaluator.mean_score())
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            if phase.lower() != "train":
                wandb_log_dict["{}/score_table".format(phase)] = wandb.Table(
                    columns=table_data[0], data=table_data[1:]
                )
            wandb.log(wandb_log_dict)
