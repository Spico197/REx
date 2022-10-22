from typing import Tuple

import torch

from rex import accelerator
from rex.metrics.base import MetricBase
from rex.tasks.simple_task import SimpleTask
from rex.utils.dict import get_dict_content
from rex.utils.io import dump_json, dump_jsonlines
from rex.utils.logging import logger
from rex.utils.progress_bar import pbar
from rex.utils.registry import register


@register("task")
class SimpleMetricTask(SimpleTask):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

    def after_initialized(self):
        super().after_initialized()

        logger.debug("Init metric")
        self.metric = self.init_metric()

    def init_metric(self) -> MetricBase:
        raise NotImplementedError

    @torch.no_grad()
    def eval(
        self, dataset_name, verbose=False, dump=False, postfix=""
    ) -> Tuple[float, dict]:
        """Eval on specific dataset and return loss and measurements

        Args:
            dataset_name: which dataset to evaluate
            verbose: whether to log evaluation results
            dump: if True, dump result to this filepath
            postfix: filepath postfix for dumping

        Returns:
            eval_loss: float
            metrics: dict
        """
        self.model.eval()
        eval_loader = self.get_data_loader(
            dataset_name, is_eval=True, epoch=self.history["curr_epoch"]
        )
        loader = pbar(
            eval_loader, desc=f"{dataset_name} - {postfix} Eval", ncols=80, ascii=True
        )

        eval_loss = 0.0
        tot_batch_results = []
        for batch in loader:
            out = self.model(**batch)
            eval_loss += out["loss"].item()
            batch_results: dict = self.metric(batch, out)
            batch_metric_score = get_dict_content(
                batch_results["metric_scores"], self.config.best_metric_field
            )
            loader.set_postfix({self.config.best_metric_field: batch_metric_score})

            tot_batch_results.append(batch_results)

        logger.info(loader)
        measurements = self.metric.get_results()

        if verbose:
            logger.info(f"Eval dataset: {dataset_name}")
            logger.info(f"Eval loss: {eval_loss}")
            logger.info(
                f"Eval metrics: {get_dict_content(measurements, self.config.best_metric_field)}"
            )
        if dump:
            dump_obj = {
                "dataset_name": dataset_name,
                "eval_loss": eval_loss,
                "metrics": measurements,
            }
            filename_prefix = (
                f"{dataset_name}.{postfix}" if len(postfix) > 0 else f"{dataset_name}"
            )
            dump_json(dump_obj, self.measures_path.joinpath(f"{filename_prefix}.json"))
            dump_jsonlines(
                tot_batch_results, self.middle_path.joinpath(f"{filename_prefix}.jsonl")
            )

        self.metric.reset()

        return eval_loss, measurements
