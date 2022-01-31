import os.path as osp
import time
import json
import logging
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
from terminaltables.ascii_table import AsciiTable
from typing import Optional
from calibrate.net import ModelWithTemperature
from calibrate.evaluation import (
    AverageMeter, LossMeter, ClassificationEvaluator, CalibrateEvaluator, OODEvaluator
)
from calibrate.utils import (
    load_train_checkpoint, load_checkpoint, save_checkpoint, round_dict
)
from calibrate.utils.torch_helper import to_numpy, get_lr
from .tester import Tester

logger = logging.getLogger(__name__)


class OODTester(Tester):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def build_data_loader(self) -> None:
        # data pipeline
        self.in_test_loader = instantiate(self.cfg.data.object.in_dist)
        self.out_test_loader = instantiate(self.cfg.data.object.out_dist)

    def build_meter(self):
        self.batch_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        self.evaluator = OODEvaluator(self.num_classes)
        self.calibrate_evaluator = CalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
            device=self.device,
        )

    def reset_meter(self):
        self.batch_time_meter.reset()
        self.evaluator.reset()
        self.calibrate_evaluator.reset()

    @torch.no_grad()
    def eval_epoch(
        self,
        phase="Val",
        temp=1.0,
        post_temp=False
    ) -> None:
        self.reset_meter()
        self.model.eval()

        end = time.time()
        for i, (inputs, labels) in enumerate(self.in_test_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            outputs = self.model(inputs)
            if post_temp:
                outputs = outputs / temp
            # metric
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                to_numpy(predicts), to_numpy(labels),
                in_dist=True
            )
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)

        for i, (inputs, labels) in enumerate(self.out_test_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            outputs = self.model(inputs)
            if post_temp:
                outputs = outputs / temp
            # metric
            predicts = F.softmax(outputs, dim=1)
            self.evaluator.update(
                to_numpy(predicts), to_numpy(labels),
                in_dist=False
            )
            self.calibrate_evaluator.update(outputs, labels)
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            end = time.time()
        self.log_eval_epoch_info(phase)

    def log_eval_epoch_info(self, phase="Val"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        metric, table_data = self.evaluator.mean_score(print=False)
        log_dict.update(metric)
        calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score(print=False)
        log_dict.update(calibrate_metric)
        logger.info("{} Epoch\t{}".format(
            phase, json.dumps(round_dict(log_dict))
        ))
        logger.info("\n" + AsciiTable(table_data).table)
        logger.info("\n" + AsciiTable(calibrate_table_data).table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict["{}/classify_score_table".format(phase)] = (
                wandb.Table(
                    columns=table_data[0],
                    data=table_data[1:]
                )
            )
            wandb_log_dict["{}/calibrate_score_table".format(phase)] = (
                wandb.Table(
                    columns=calibrate_table_data[0],
                    data=calibrate_table_data[1:]
                )
            )
            wandb.log(wandb_log_dict)

    def test(self):
        logger.info(
            "Everything is perfect so far. Let's start testing. Good luck!"
        )
        self.eval_epoch(phase="Test")
        if self.cfg.post_temperature.enable:
            logger.info("Test with post-temperature scaling!")
            temp = self.post_temperature()
            self.eval_epoch(phase="TestPT", temp=temp, post_temp=True)
        # temp = self.post_temperature()
        # self.eval_epoch(phase="TestPT", temp=temp, post_temp=True)