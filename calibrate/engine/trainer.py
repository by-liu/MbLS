import os.path as osp
from shutil import copyfile
import time
import json
import logging
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import wandb
from terminaltables.ascii_table import AsciiTable

from calibrate.net import ModelWithTemperature
from calibrate.losses import LogitMarginL1
from calibrate.evaluation import (
    AverageMeter, LossMeter, ClassificationEvaluator,
    CalibrateEvaluator, LogitsEvaluator, ProbsEvaluator
)
from calibrate.utils import (
    load_train_checkpoint, load_checkpoint, save_checkpoint, round_dict
)
from calibrate.utils.torch_helper import to_numpy, get_lr

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.work_dir = self.cfg.work_dir
        self.device = torch.device(self.cfg.device)
        self.build_data_loader()
        self.build_model()
        self.build_solver()
        self.build_meter()
        self.init_wandb_or_not()

    def build_data_loader(self) -> None:
        # data pipeline
        self.train_loader, self.val_loader = instantiate(self.cfg.data.object.trainval)
        logger.info("Data pipeline initialized")

    def build_model(self) -> None:
        # network
        self.model = instantiate(self.cfg.model.object)
        self.model.to(self.device)
        self.loss_func = instantiate(self.cfg.loss.object)
        self.loss_func.to(self.device)
        logger.info(self.loss_func)
        logger.info("Model initialized")

    def build_solver(self) -> None:
        # build solver
        self.optimizer = instantiate(
            self.cfg.optim.object, self.model.parameters()
        )
        self.scheduler = instantiate(
            self.cfg.scheduler.object, self.optimizer
        )
        logger.info("Solver initialized")

    def init_wandb_or_not(self) -> None:
        if self.cfg.wandb.enable:
            wandb.init(
                project=self.cfg.wandb.project,
                entity=self.cfg.wandb.entity,
                config=OmegaConf.to_container(self.cfg, resolve=True),
                tags=["train"],
            )
            wandb.run.name = "{}-{}-{}".format(
                wandb.run.id, self.cfg.model.name, self.cfg.loss.name
            )
            wandb.run.save()
            wandb.watch(self.model, log=None)
            logger.info("Wandb initialized : {}".format(wandb.run.name))

    def start_or_resume(self):
        if self.cfg.train.resume:
            self.start_epoch, self.best_epoch, self.best_score = (
                load_train_checkpoint(
                    self.work_dir, self.device, self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler
                )
            )
        else:
            self.start_epoch, self.best_epoch, self.best_score = 0, -1, None
        self.max_epoch = self.cfg.train.max_epoch

    def build_meter(self):
        self.batch_time_meter = AverageMeter()
        self.data_time_meter = AverageMeter()
        self.num_classes = self.cfg.model.num_classes
        if hasattr(self.loss_func, "names"):
            self.loss_meter = LossMeter(
                num_terms=len(self.loss_func.names),
                names=self.loss_func.names
            )
        else:
            self.loss_meter = LossMeter()
        self.evaluator = ClassificationEvaluator(self.num_classes)
        self.calibrate_evaluator = CalibrateEvaluator(
            self.num_classes,
            num_bins=self.cfg.calibrate.num_bins,
            device=self.device,
        )
        self.logits_evaluator = LogitsEvaluator()
        # self.probs_evaluator = ProbsEvaluator(self.num_classes)

    def reset_meter(self):
        self.batch_time_meter.reset()
        self.data_time_meter.reset()
        self.loss_meter.reset()
        self.evaluator.reset()
        self.calibrate_evaluator.reset()
        self.logits_evaluator.reset()

    def log_iter_info(self, iter, max_iter, epoch, phase="Train"):
        log_dict = {}
        log_dict["data_time"] = self.data_time_meter.val
        log_dict["batch_time"] = self.batch_time_meter.val
        log_dict.update(self.loss_meter.get_vals())
        log_dict.update(self.evaluator.curr_score())
        log_dict.update(self.logits_evaluator.curr_score())
        # log_dict.update(self.probs_evaluator.curr_score())
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
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict["lr"] = get_lr(self.optimizer)
        log_dict.update(self.loss_meter.get_avgs())
        if isinstance(self.loss_func, LogitMarginL1):
            log_dict["alpha"] = self.loss_func.alpha
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

    def log_eval_epoch_info(self, epoch, phase="Val"):
        log_dict = {}
        log_dict["samples"] = self.evaluator.num_samples()
        log_dict.update(self.loss_meter.get_avgs())
        classify_metric, classify_table_data = self.evaluator.mean_score(print=False)
        log_dict.update(classify_metric)
        calibrate_metric, calibrate_table_data = self.calibrate_evaluator.mean_score(print=False)
        log_dict.update(calibrate_metric)
        log_dict.update(self.logits_evaluator.mean_score())
        # log_dict.update(self.probs_evaluator.mean_score())
        logger.info("{} Epoch[{}]\t{}".format(
            phase, epoch + 1, json.dumps(round_dict(log_dict))
        ))
        logger.info("\n" + AsciiTable(classify_table_data).table)
        logger.info("\n" + AsciiTable(calibrate_table_data).table)
        if self.cfg.wandb.enable:
            wandb_log_dict = {"epoch": epoch}
            wandb_log_dict.update(dict(
                ("{}/{}".format(phase, key), value) for (key, value) in log_dict.items()
            ))
            wandb_log_dict["{}/classify_score_table".format(phase)] = (
                wandb.Table(
                    columns=classify_table_data[0],
                    data=classify_table_data[1:]
                )
            )
            wandb_log_dict["{}/calibrate_score_table".format(phase)] = (
                wandb.Table(
                    columns=calibrate_table_data[0],
                    data=calibrate_table_data[1:]
                )
            )
            if "test" in phase.lower() and self.cfg.calibrate.visualize:
                fig_reliab, fig_hist = self.calibrate_evaluator.plot_reliability_diagram()
                wandb_log_dict["{}/calibrate_reliability".format(phase)] = fig_reliab
                wandb_log_dict["{}/confidence_histogram".format(phase)] = fig_hist
            wandb.log(wandb_log_dict)

    def train_epoch(self, epoch: int):
        self.reset_meter()
        self.model.train()

        max_iter = len(self.train_loader)

        end = time.time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            # compute the time for data loading
            self.data_time_meter.update(time.time() - end)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            outputs = self.model(inputs)
            loss = self.loss_func(outputs, labels)
            if isinstance(loss, tuple):
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
            predicts = F.softmax(outputs, dim=1)
            # pred_labels = torch.argmax(predicts, dim=1)
            self.evaluator.update(
                to_numpy(predicts), to_numpy(labels)
            )
            self.logits_evaluator.update(to_numpy(outputs))
            # self.probs_evaluator.update(to_numpy(predicts))
            # measure elapsed time
            self.batch_time_meter.update(time.time() - end)
            if (i + 1) % self.cfg.log_period == 0:
                self.log_iter_info(i, max_iter, epoch)
            end = time.time()
        self.log_epoch_info(epoch)

    @torch.no_grad()
    def eval_epoch(
        self, data_loader, epoch,
        phase="Val",
        temp=1.0,
        post_temp=False
    ):
        self.reset_meter()
        self.model.eval()

        max_iter = len(data_loader)
        end = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # forward
            outputs = self.model(inputs)
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
            val_loss, val_score = self.eval_epoch(self.val_loader, epoch, phase="Val")
            # run lr scheduler
            self.scheduler.step()
            if isinstance(self.loss_func, LogitMarginL1):
                self.loss_func.schedule_alpha(epoch)
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
                keep_checkpoint_num=self.cfg.train.keep_checkpoint_num,
                keep_checkpoint_interval=self.cfg.train.keep_checkpoint_interval
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
        model_with_temp = ModelWithTemperature(
            self.model,
            learn=self.cfg.post_temperature.learn,
            grid_search_interval=self.cfg.post_temperature.grid_search_interval,
            cross_validate=self.cfg.post_temperature.cross_validate,
            device=self.device
        )
        model_with_temp.set_temperature(self.val_loader)
        temp = model_with_temp.get_temperature()
        wandb.log({
            "temperature": temp
        })
        return temp

    def test(self):
        logger.info("We are almost done : final testing ...")
        self.test_loader = instantiate(self.cfg.data.object.test)
        # test best pth
        epoch = self.best_epoch
        logger.info("#################")
        logger.info(" Test at best epoch {}".format(epoch + 1))
        logger.info("#################")
        logger.info("Best epoch[{}] :".format(epoch + 1))
        load_checkpoint(
            osp.join(self.work_dir, "best.pth"), self.model, self.device
        )
        self.eval_epoch(self.test_loader, epoch, phase="Test")
        if self.cfg.post_temperature.enable:
            logger.info("Test with post-temperature scaling!")
            temp = self.post_temperature()
            self.eval_epoch(self.test_loader, epoch, phase="TestPT", temp=temp, post_temp=True)

    def run(self):
        self.train()
        self.test()
