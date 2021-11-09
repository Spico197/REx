import os
import json
import logging
from typing import Optional
from omegaconf import OmegaConf

import torch
from torch import distributed
from torch.nn import parallel
from loguru import logger


class TaskBase(object):
    def __init__(self, config: OmegaConf) -> None:
        self.config = config
        self.model = None
        self.optimizer = None

        self.history = {"train": [], "dev": [], "test": []}
        self.no_climbing_cnt = 0
        self.best_metric = -100.0
        self.best_epoch = -1

        if not os.path.exists(config.task_dir):
            os.makedirs(config.task_dir)
        else:
            logger.warning(f"Overwrite task dir: {config.task_dir}")
        OmegaConf.save(config, os.path.join(config.task_dir, "config.yaml"))

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        raise NotImplementedError

    def load(
        self,
        path: str,
        load_model: Optional[bool] = True,
        load_optimizer: Optional[bool] = False,
    ):
        if os.path.exists(path):
            logger.info("Resume checkpoint from {}".format(path))
        else:
            raise ValueError("Checkpoint does not exist, {}".format(path))

        if torch.cuda.device_count() == 0:
            store_dict = torch.load(path, map_location="cpu")
        else:
            store_dict = torch.load(path, map_location=torch.device(self.config.device))

        self.config = OmegaConf.load(os.path.join(self.config.task_dir, "config.yaml"))

        if load_model:
            if self.model and "model_state" in store_dict:
                if isinstance(self.model, parallel.DataParallel) or isinstance(
                    self.model, parallel.DistributedDataParallel
                ):
                    self.model.module.load_state_dict(store_dict["model_state"])
                else:
                    self.model.load_state_dict(store_dict["model_state"])
                logger.info("Load model successfully")
            else:
                raise ValueError(
                    f"Model loading failed. self.model={self.model}, stored_dict_keys={store_dict.keys()}"
                )
        else:
            logger.info("Not load model")

        if load_optimizer:
            if self.optimizer and "optimizer_state" in store_dict:
                self.optimizer.load_state_dict(store_dict["optimizer_state"])
                logger.info("Load optimizer successfully")
            else:
                raise ValueError(
                    f"Model loading failed. self.model={self.optimizer}, stored_dict_keys={store_dict.keys()}"
                )
        else:
            logger.info("Not load optimizer")

        self.history = store_dict["history"]
        self.no_climbing_cnt = store_dict["no_climbing_cnt"]
        self.best_metric = store_dict["best_metric"]
        self.best_epoch = store_dict["best_epoch"]

    def save(self, path, epoch: Optional[int] = None):
        logger.info(f"Dumping checkpoint into: {path}")
        store_dict = {}

        if self.model:
            if isinstance(self.model, parallel.DataParallel) or isinstance(
                self.model, parallel.DistributedDataParallel
            ):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            store_dict["model_state"] = model_state
        else:
            logger.info("No model state is dumped", logging.WARNING)

        if self.optimizer:
            store_dict["optimizer_state"] = self.optimizer.state_dict()
        else:
            logger.info("No optimizer state is dumped", logging.WARNING)

        if epoch:
            store_dict["epoch"] = epoch

        store_dict["history"] = self.history
        store_dict["no_climbing_cnt"] = self.no_climbing_cnt
        store_dict["best_metric"] = self.best_metric
        store_dict["best_epoch"] = self.best_epoch

        torch.save(store_dict, path)

    def save_ckpt(self, identifier, epoch: Optional[int] = None):
        ckpt_dir = os.path.join(self.config.task_dir, "ckpt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_name = f"{self.model.__class__.__name__}.{identifier}.pth"
        self.save(os.path.join(ckpt_dir, ckpt_name), epoch)

    def logging(self, msg: str, level: Optional[int] = logging.INFO):
        if self.in_distributed_mode():
            msg = "Rank {} {}".format(distributed.get_rank(), msg)
        if self.config.only_master_logging:
            if self.is_master_node():
                logger.log(level, msg)
        else:
            logger.log(level, msg)

    def is_master_node(self):
        if self.in_distributed_mode():
            if distributed.get_rank() == 0:
                return True
            else:
                return False
        else:
            return True

    def in_distributed_mode(self):
        return self.config.local_rank >= 0
