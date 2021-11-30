import numpy as np

from calibrate.utils.constants import EPS
from .evaluator import DatasetEvaluator


class ProbsEvaluator(DatasetEvaluator):
    """get probs (softmax output) statics
    max_probs / confidence : probs.max()
    mean_probs : probs.mean()
    kl_div : kl divergence between probs and 1/num_classes
    l1_div : l1 between probs and 1/num_classes
    """

    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes
        self.reset()

    def reset(self) -> None:
        self.count = 0
        self.max_probs = []
        self.mean_probs = []
        self.kl_divs = []
        self.l1_divs = []

    def num_samples(self):
        return self.count

    def main_metric(self):  
        return "max_prob"

    def kl(self, probs):
        y = np.log(1 / self.num_classes) - np.log(probs + EPS)
        y = np.mean(y, axis=1)
        return y

    def l1(self, probs):
        y = np.abs(1 / self.num_classes - probs)
        y = np.mean(y, axis=1)
        return y

    def update(self, probs: np.ndarray):
        n = probs.shape[0]
        self.count += n

        max_probs = np.max(probs, axis=1)
        mean_probs = np.mean(probs, axis=1)
        kl_divs = self.kl(probs)
        l1_divs = self.l1(probs)

        self.max_probs.append(max_probs)
        self.mean_probs.append(mean_probs)
        self.kl_divs.append(kl_divs)
        self.l1_divs.append(l1_divs)

        return float(np.mean(max_probs))

    def curr_score(self):
        return {self.main_metric(): float(np.mean(self.max_probs[-1]))}

    def mean_score(self, all_metric=True):
        max_probs = np.concatenate(self.max_probs)
        mean_probs = np.concatenate(self.mean_probs)
        kl_divs = np.concatenate(self.kl_divs)
        l1_divs = np.concatenate(self.l1_divs)

        if not all_metric:
            return np.mean(max_probs)

        metric = {}
        metric["max_prob"] = float(np.mean(max_probs))
        metric["mean_prob"] = float(np.mean(mean_probs))
        metric["kl_div"] = float(np.mean(kl_divs))
        metric["l1_div"] = float(np.mean(l1_divs))

        return metric
