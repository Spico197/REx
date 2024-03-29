from typing import Optional, Tuple

from rex.utils.deprecation import deprecation_warning


class MetricBase(object):
    def __init__(self) -> None:
        # prediction history for calculating metric scores
        self.preds = []
        # gold history for calculating metric scores
        self.golds = []

    def reset(self):
        """
        Clear history, often used in the end of one epoch's evaluation
        """
        self.preds.clear()
        self.golds.clear()

    def compute(self, golds: Optional[list] = None, preds: Optional[list] = None):
        if (golds is None) ^ (preds is None):
            raise ValueError(
                "Please make sure golds and preds are both None or both not None"
            )

        if golds and preds:
            return self.calculate_scores(golds, preds)
        else:
            return self.calculate_scores(self.golds, self.preds)

    def get_results(self):
        """Get evaluation results from history"""
        deprecation_warning("get_results", "compute")
        return self.calculate_scores(self.golds, self.preds)

    def add_batch(self, raw_batch: dict, out_batch: dict) -> dict:
        """
        Calculate evaluation metric scores of current batch
            and add pred & gold results to history.

        Args:
            raw_batch: raw input batch from data loader
            out_batch: predicted results of current batch from model

        Returns:
            a dict containing neccessary information from
        """
        gold_instances, pred_instances = self.get_instances_from_batch(
            raw_batch, out_batch
        )

        self.golds.extend(gold_instances)
        self.preds.extend(pred_instances)

        batch_result = {
            "gold": gold_instances,
            "pred": pred_instances,
            "metric_scores": self.calculate_scores(gold_instances, pred_instances),
        }

        return batch_result

    def __call__(self, raw_batch: dict, out_batch: dict) -> dict:
        return self.add_batch(raw_batch, out_batch)

    def get_instances_from_batch(self, raw_batch: dict, out_batch: dict) -> Tuple:
        """
        Process raw batch input and predicted output into
            gold & pred instances

        Args:
            raw_batch: raw input batch from data loader
            out_batch: predicted results of current batch from model

        Returns:
            gold_instances: list of instances
            pred_instances: list of pred instances
        """
        raise NotImplementedError

    def calculate_scores(self, golds: list, preds: list) -> dict:
        """
        Calculate scores from gold and pred instances

        Args:
            golds: a list of gold instances obtained from `get_instances_from_batch()`
            preds: a list of predicted instances

        Returns:
            a dict of metric scores
        """
        raise NotImplementedError
