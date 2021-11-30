import logging
import numpy as np
from terminaltables import AsciiTable
import pandas as pd
from typing import List, Optional
import wandb

from .evaluator import DatasetEvaluator
from calibrate.utils.constants import EPS

logger = logging.getLogger(__name__)


def intersect_and_union(pred_label, label, num_classes, ignore_index):
    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect, _ = np.histogram(
        intersect, bins=np.arange(num_classes + 1)
    )
    area_pred_label, _ = np.histogram(
        pred_label, bins=np.arange(num_classes + 1)
    )
    area_label, _ = np.histogram(
        label, bins=np.arange(num_classes + 1)
    )
    area_union = area_pred_label + area_label - area_intersect

    return area_intersect, area_union, area_pred_label, area_label


class SegmentEvaluator(DatasetEvaluator):
    def __init__(self,
                 classes: Optional[List[str]] = None,
                 ignore_index: int = -1) -> None:
        super().__init__()
        self.classes = classes
        self.num_classes = len(self.classes)
        self.ignore_index = ignore_index

    def num_samples(self):
        return self.nsamples

    def reset(self):
        self.total_area_inter = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_union = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_pred = np.zeros((self.num_classes, ), dtype=np.float)
        self.total_area_target = np.zeros((self.num_classes, ), dtype=np.float)
        self.nsamples = 0

    def main_metric(self):
        return "miou"

    def ignore_background(self, pred: np.ndarray, target: np.ndarray):
        pred = pred[:, 1:] if pred.shape[1] > 1 else pred
        target = target[:, 1:] if target.shape[1] > 1 else target
        return pred, target

    def update(self, pred: np.ndarray, target: np.ndarray):
        """Update all the metric from batch size prediction and target.

        Args:
            pred: predictions to be evaluated in one-hot formation
            y: ground truth. It should be one-hot format.
        """
        assert pred.shape == target.shape, "pred and target should have same shapes"
        n = pred.shape[0]
        self.nsamples += n

        batch_area_inter = np.zeros((self.num_classes, ), dtype=np.float)
        batch_area_union = np.zeros((self.num_classes, ), dtype=np.float)
        batch_area_pred = np.zeros((self.num_classes, ), dtype=np.float)
        batch_area_target = np.zeros((self.num_classes, ), dtype=np.float)
        for i in range(n):
            area_inter, area_union, area_pred, area_target = (
                intersect_and_union(
                    pred[i], target[i], self.num_classes, self.ignore_index
                )
            )
            batch_area_inter += area_inter
            batch_area_union += area_union
            batch_area_pred += area_pred
            batch_area_target += area_target

        iou = batch_area_inter[1:].sum() / (batch_area_union[1:].sum() + EPS)
        self.curr = {"iou": iou}

        # update the total
        self.total_area_inter += batch_area_inter
        self.total_area_union += batch_area_union
        self.total_area_pred += batch_area_pred
        self.total_area_target += batch_area_target

    def curr_score(self):
        return self.curr

    def mean_score(self, main=False):
        mdice = (
            2 * self.total_area_inter[1:]
            / (self.total_area_pred[1:] + self.total_area_target[1:] + EPS)
        ).mean()
        miou = (
            self.total_area_inter[1:] / (self.total_area_union[1:] + EPS)
        ).mean()
        macc = (
            self.total_area_inter[1:] / (self.total_area_target[1:] + EPS)
        ).mean()
        if main:
            return miou
        else:
            return {"mdsc": mdice, "miou": miou, "macc": macc}

    def class_score(self, print=True, return_dataframe=False):
        class_acc = self.total_area_inter[1:] / (self.total_area_target[1:] + EPS)
        class_dice = (
            2 * self.total_area_inter[1:]
            / (self.total_area_pred[1:] + self.total_area_target[1:] + EPS)
        )
        class_iou = self.total_area_inter[1:] / (self.total_area_union[1:] + EPS)
        columns = ["id", "Class", "iou", "dsc", "acc"]
        class_table_data = [columns]
        for i in range(class_acc.shape[0]):
            class_table_data.append(
                [i] + [self.classes[i + 1]]
                + ["{:.4f}".format(class_iou[i])]
                + ["{:.4f}".format(class_dice[i])]
                + ["{:.4f}".format(class_acc[i])]
            )
        class_table_data.append(
            [""] + ["mean"]
            + ["{:.4f}".format(np.mean(class_iou))]
            + ["{:.4f}".format(np.mean(class_dice))]
            + ["{:.4f}".format(np.mean(class_acc))]
        )

        if print:
            table = AsciiTable(class_table_data)
            logger.info("\n" + table.table)

        if return_dataframe:
            data = {key: [] for key in columns}
            for i in range(class_acc.shape[0]):
                data[columns[0]].append(i)
                data[columns[1]].append(self.classes[i + 1])
                data[columns[3]].append(class_iou[i])
                data[columns[2]].append(class_dice[i])
                data[columns[4]].append(class_acc[i])
            data[columns[0]].append(None)
            data[columns[1]].append("mean")
            data[columns[3]].append(np.mean(class_iou))
            data[columns[2]].append(np.mean(class_dice))
            data[columns[4]].append(np.mean(class_acc))
            return pd.DataFrame(data, columns=columns)

    def wandb_score_table(self):
        table_data = self.class_score(print=False, return_dataframe=True)
        return wandb.Table(dataframe=table_data)
