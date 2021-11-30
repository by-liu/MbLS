import math
import time
import os.path as osp
from shutil import copyfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import wandb
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from calibrate.engine.trainer import Trainer
from calibrate.net import ModelWithTemperature
from calibrate.losses import LabelSmoothConstrainedLoss
from calibrate.utils.torch_helper import to_numpy
from calibrate.utils import (
    load_train_checkpoint, load_checkpoint, save_checkpoint, round_dict
)

logger = logging.getLogger(__name__)


class NLPTrainer(Trainer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def build_data_loader(self) -> None:
        (
            self.embedding_matrix,
            self.train_datas,
            self.train_labels,
            self.val_datas,
            self.val_labels,
            self.test_datas,
            self.test_labels,
            self.num_words,
            self.embedding_dim
        ) = instantiate(self.cfg.data.object.all)
        self.batch_size = self.cfg.data.batch_size
        self.num_classes = 20

    def build_model(self) -> None:
        # embedding
        self.embedding_model = nn.Embedding(self.num_words, self.embedding_dim)
        self.embedding_model.to(self.device)
        self.embedding_model.state_dict()["weight"].copy_(self.embedding_matrix)
        # network
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        # loss
        self.loss_func = instantiate(self.cfg.loss.object)
        self.loss_func.to(self.device)
        logger.info(self.loss_func)
        logger.info("Model initialized")

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()
        self.embedding_model.eval()

        perm = np.random.permutation(np.arange(len(self.train_datas)))
        perm_train = np.take(self.train_datas, perm, axis=0)
        perm_labels = np.take(self.train_labels, perm, axis=0)
        max_iter = perm_train.shape[0] // self.batch_size

        end = time.time()
        for i in range(max_iter):
            inputs = torch.from_numpy(
                perm_train[i * self.batch_size:(i + 1) * self.batch_size]
            ).type(torch.LongTensor).to(self.device)
            labels = torch.from_numpy(
                np.argmax(perm_labels[i * self.batch_size:(i + 1) * self.batch_size], 1)
            ).to(self.device)
            self.data_time_meter.update(time.time() - end)

            with torch.no_grad():
                embs = self.embedding_model(inputs)
            outputs = self.model(embs)
            loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
                loss_total = loss[0]
            else:
                loss_total = loss
            # backward
            self.optimizer.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # metric
            self.loss_meter.update(loss, inputs.size(0))
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                to_numpy(predicts), to_numpy(labels)
            )
            self.logits_evaluator.update(to_numpy(outputs))
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_epoch_info(epoch)

    @torch.no_grad()
    def eval_epoch(
        self, eval_data, eval_labels, epoch,
        phase="Val",
        temp=1,
        post_temp=False
    ):
        self.reset_meter()
        self.model.eval()
        self.embedding_model.eval()

        max_iter = math.ceil(eval_data.shape[0] // self.batch_size)

        end = time.time()
        for i in range(max_iter):
            inputs = torch.from_numpy(
                eval_data[i * self.batch_size:min((i + 1) * self.batch_size, eval_data.shape[0])]
            ).type(torch.LongTensor).to(self.device)
            labels = torch.from_numpy(
                np.argmax(eval_labels[i * self.batch_size:min((i+1) * self.batch_size, eval_data.shape[0])], 1)
            ).to(self.device)
            embs = self.embedding_model(inputs)
            outputs = self.model(embs)
            if post_temp:
                outputs = outputs / temp
            loss = self.loss_func(outputs, labels)
            # metric
            self.loss_meter.update(loss)
            self.calibrate_evaluator.update(outputs, labels)
            self.logits_evaluator.update(to_numpy(outputs))
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                to_numpy(predicts), to_numpy(labels)
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            # logging
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch, phase)
            end = time.time()
        self.log_eval_epoch_info(epoch, phase)

        return self.loss_meter.avg(0), self.evaluator.mean_score(all_metric=False)[0]

    def train(self):
        self.start_or_resume()
        logger.info(
            "Everything is perfect so far. Let's start training. Good luck!"
        )
        for epoch in range(self.start_epoch, self.max_epoch):
            logger.info("=" * 20)
            logger.info(" Start epoch {}".format(epoch + 1))
            logger.info("=" * 20)
            self.train_epoch(epoch)
            val_loss, val_score = self.eval_epoch(self.val_datas, self.val_labels, epoch, phase="Val")
            # run lr scheduler
            self.scheduler.step()
            if isinstance(self.loss_func, LabelSmoothConstrainedLoss):
                self.loss_func.schedule_t()
            if self.best_score is None or val_score > self.best_score:
                self.best_score, self.best_epoch = val_score, epoch
                best_checkpoint = True
            else:
                best_checkpoint = False
            save_checkpoint(
                self.work_dir, self.model, self.optimizer, self.scheduler,
                epoch=epoch,
                best_checkpoint=best_checkpoint,
                val_score=val_score,
                keep_checkpoint_num=self.cfg.train.keep_checkpoint_num
            )
            # logging best performance on val so far
            logger.info(
                "Epoch[{}]\tBest {} on Val : {:.4f} at epoch {}".format(
                    epoch + 1, self.evaluator.main_metric(),
                    self.best_score, self.best_epoch + 1
                )
            )
            if self.cfg.wandb.enable and best_checkpoint:
                wandb.log({
                    "epoch": epoch,
                    "Val/best_epoch": self.best_epoch,
                    "Val/best_{}".format(self.evaluator.main_metric()): self.best_score,
                    "Val/best_classify_score_table": self.evaluator.wandb_score_table(),
                    "Val/best_calibrate_score_table": self.calibrate_evaluator.wandb_score_table()
                })
        if self.cfg.wandb.enable:
            copyfile(
                osp.join(self.work_dir, "best.pth"),
                osp.join(self.work_dir, "{}-best.pth".format(wandb.run.name))
            )

    def post_temperature(self):
        model_with_temp = ModelWithTemperature(self.model, device=self.device)
        model_with_temp.set_temperature_ng(
            self.embedding_model, self.val_datas, self.val_labels,
            batch_size=self.batch_size
        )
        temp = model_with_temp.get_temperature()
        wandb.log({
            "temperature": temp
        })
        return temp

    def test(self):
        logger.info("We are almost done : final testing ...")
        # test best pth
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(" Test at best epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Best epoch[{}] :".format(epoch + 1))
        load_checkpoint(
            osp.join(self.work_dir, "best.pth"), self.model, self.device
        )
        self.eval_epoch(self.test_datas, self.test_labels, epoch, phase="Test")
        temp = self.post_temperature()
        self.eval_epoch(self.test_datas, self.test_labels, epoch, phase="TestPT", temp=temp, post_temp=True)
