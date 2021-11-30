import logging
from terminaltables import AsciiTable
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import wandb
from sklearn.metrics import top_k_accuracy_score, roc_auc_score

from .evaluator import DatasetEvaluator
from .metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss
from .reliability_diagram import ReliabilityDiagram
from calibrate.utils.torch_helper import to_numpy
from calibrate.utils.constants import EPS

logger = logging.getLogger(__name__)

class OODEvaluator(DatasetEvaluator):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.in_preds = None
        self.in_labels = None
        self.out_preds = None
        self.out_labels = None

    def main_metric(self):
        return "auc_ent"

    def num_samples(self):
        return (
            (
                self.in_labels.shape[0]
                if self.in_labels is not None
                else 0
            )
            + (
                self.out_labels.shape[0]
                if self.out_labels is not None
                else 0
            )
        )

    def _update(self, all_preds, all_labels, pred, label):
        if all_preds is None:
            all_preds = pred
            all_labels = label
        else:
            all_preds = np.concatenate((all_preds, pred), axis=0)
            all_labels = np.concatenate((all_labels, label), axis=0)

    def update(self, pred: np.ndarray, label: np.ndarray,
               in_dist: bool = True) -> float:
        """update

        Args:
            pred (np.ndarray): n x num_classes
            label (np.ndarray): n x 1

        Returns:
            float: acc
        """
        assert pred.shape[0] == label.shape[0]
        if in_dist:
            if self.in_preds is None:
                self.in_preds = pred
                self.in_labels = label
            else:
                self.in_preds = np.concatenate((self.in_preds, pred), axis=0)
                self.in_labels = np.concatenate((self.in_labels, label), axis=0)
        else:
            if self.out_preds is None:
                self.out_preds = pred
                self.out_labels = label
            else:
                self.out_preds = np.concatenate((self.out_preds, pred), axis=0)
                self.out_labels = np.concatenate((self.out_labels, label), axis=0)

        pred_label = np.argmax(pred, axis=1)
        acc = (pred_label == label).astype("int").sum() / label.shape[0]

        # acc = top_k_accuracy_score(label, pred, k=1)

        self.curr = {"acc": acc}
        return acc

    def curr_score(self):
        return self.curr

    def entropy(self, preds):
        log_preds = np.log(preds + EPS)
        entropies = - np.sum(preds * log_preds, axis=1) / np.log(self.num_classes)
        return entropies

    def mean_score(self, print=False, all_metric=True):
        # acc = top_k_accuracy_score(self.labels, self.preds, k=1)

        in_labels_entropies = np.zeros(self.in_labels.shape)
        in_preds_entropies = self.entropy(self.in_preds)
        out_labels_entropies = np.ones(self.out_labels.shape)
        out_preds_entropies = self.entropy(self.out_preds)

        labels_entropies = np.concatenate(
            (in_labels_entropies, out_labels_entropies),
            axis=0
        )
        preds_entropies = np.concatenate(
            (in_preds_entropies, out_preds_entropies),
            axis=0
        )

        in_labels_confidences = np.ones(self.in_labels.shape)
        in_preds_confidences = np.max(self.in_preds, axis=1)
        out_labels_confidences = np.zeros(self.out_labels.shape)
        out_preds_confidences = np.max(self.out_preds, axis=1)

        labels_confidences = np.concatenate(
            (in_labels_confidences, out_labels_confidences),
            axis=0
        )
        preds_confidences = np.concatenate(
            (in_preds_confidences, out_preds_confidences),
            axis=0
        )


        auc_ent = roc_auc_score(labels_entropies, preds_entropies)
        auc_conf = roc_auc_score(labels_confidences, preds_confidences)

        metric = {"auc_ent": auc_ent, "auc_conf": auc_conf}

        columns = ["samples", "auc_ent", "auc_conf"]
        table_data = [columns]
        table_data.append(
            [
                self.num_samples(),
                "{:.5f}".format(auc_ent),
                "{:.5f}".format(auc_conf)
            ]
        )

        if print:
            table = AsciiTable(table_data)
            logger.info("\n" + table.table)

        if all_metric:
            return metric, table_data
        else:
            return metric[self.main_metric()], table_data

    def wandb_score_table(self):
        _, table_data = self.mean_score(print=False)
        return wandb.Table(
            columns=table_data[0],
            data=table_data[1:]
        )
