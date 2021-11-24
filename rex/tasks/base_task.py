import os
import logging
from pathlib import Path
from typing import Optional

import torch
from torch import distributed
from torch.nn import parallel
from loguru import logger
from omegaconf import OmegaConf

from rex.utils.initialization import init_all


CONFIG_PARAMS_FILENAME = "task_params.yaml"
CHECKPOINT_DIRNAME = "ckpt"
CHECKPOINT_FILENAME_TEMPLATE = "{}.{}.pth"
BEST_IDENTIFIER = "best"
BEST_CHECKPOINT_FILENAME_TEMPLATE = "{}.best.pth"
LOG_FILENAME = "log.log"


class TaskBase(object):
    def __init__(
        self,
        config: OmegaConf,
        initialize: Optional[bool] = True,
        makedirs: Optional[bool] = True,
        dump_configfile: Optional[bool] = True,
    ) -> None:
        self.config = config
        self.model = None
        self.optimizer = None

        self.history = {"train": [], "dev": [], "test": []}
        self.no_climbing_cnt = 0
        self.best_metric = -100.0
        self.best_epoch = -1

        if initialize:
            init_all(
                config.task_dir, seed=config.random_seed, set_cudnn=True, config=None
            )

        config_string = OmegaConf.to_yaml(config, resolve=True)
        logger.info(config_string)

        if makedirs:
            if not os.path.exists(config.task_dir):
                os.makedirs(config.task_dir)
            else:
                logger.warning(f"Overwrite task dir: {config.task_dir}")

        if dump_configfile:
            OmegaConf.save(
                config,
                os.path.join(config.task_dir, CONFIG_PARAMS_FILENAME),
                resolve=True,
            )

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

        self.history = store_dict.pop("history")
        self.no_climbing_cnt = store_dict.pop("no_climbing_cnt")
        self.best_metric = store_dict.pop("best_metric")
        self.best_epoch = store_dict.pop("best_epoch")

    def load_best_ckpt(self, load_optimizer: Optional[bool] = False):
        _task_dir = Path(self.config.task_dir)
        model_name = self.model.__class__.__name__
        ckpt_filepath = str(
            _task_dir.joinpath(
                CHECKPOINT_DIRNAME,
                BEST_CHECKPOINT_FILENAME_TEMPLATE.format(model_name),
            )
        )
        logger.info(f"Loading model from: {ckpt_filepath}")
        self.load(ckpt_filepath, load_model=True, load_optimizer=load_optimizer)

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

    def save_ckpt(
        self, identifier: Optional[str] = BEST_IDENTIFIER, epoch: Optional[int] = None
    ):
        ckpt_dir = os.path.join(self.config.task_dir, CHECKPOINT_DIRNAME)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_name = CHECKPOINT_FILENAME_TEMPLATE.format(
            self.model.__class__.__name__, identifier
        )
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

    @classmethod
    def from_configfile(
        cls,
        config_filepath: str,
        load_train_data: Optional[bool] = True,
        load_dev_data: Optional[bool] = True,
        load_test_data: Optional[bool] = True,
        **kwargs,
    ):
        logger.info(f"Initializing from configuration file: {config_filepath}")
        config = OmegaConf.load(config_filepath)

        # in case of any redundant memory taken when inference
        config["load_train_data"] = load_train_data
        config["load_dev_data"] = load_dev_data
        config["load_test_data"] = load_test_data

        kwargs["initialize"] = kwargs.pop("initialize", True)
        kwargs["makedirs"] = kwargs.pop("makedirs", False)
        kwargs["dump_configfile"] = kwargs.pop("dump_configfile", False)
        return cls(config, **kwargs)

    @classmethod
    def from_taskdir(
        cls,
        task_dir: str,
        load_best_model: Optional[bool] = True,
        load_best_optimizer: Optional[bool] = False,
        load_train_data: Optional[bool] = True,
        load_dev_data: Optional[bool] = True,
        load_test_data: Optional[bool] = True,
        **kwargs,
    ):
        _task_dir = Path(task_dir)
        config_filepath = str(_task_dir.joinpath(CONFIG_PARAMS_FILENAME).absolute())
        ins = cls.from_configfile(
            config_filepath,
            load_train_data=load_train_data,
            load_dev_data=load_dev_data,
            load_test_data=load_test_data,
            **kwargs,
        )
        if load_best_model:
            ins.load_best_ckpt(load_optimizer=load_best_optimizer)
        return ins
