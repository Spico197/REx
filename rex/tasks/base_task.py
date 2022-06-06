import logging
import os
from pathlib import Path
from typing import Optional

import torch
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch import distributed
from torch.nn import parallel

from rex import accelerator
from rex.tasks import (
    BEST_CHECKPOINT_FILENAME_TEMPLATE,
    BEST_IDENTIFIER,
    CHECKPOINT_DIRNAME,
    CHECKPOINT_FILENAME_TEMPLATE,
    CONFIG_PARAMS_FILENAME,
)
from rex.utils.initialization import init_all
from rex.utils.logging import logger
from rex.utils.wrapper import rank_zero_only


class TaskBase(object):
    def __init__(
        self,
        config: OmegaConf,
        initialize: Optional[bool] = True,
        makedirs: Optional[bool] = True,
        dump_configfile: Optional[bool] = True,
    ) -> None:
        self.config = config

        # `batch_size` in config is the total batch size number
        self.device = accelerator.device

        self.model = None
        self.optimizer = None

        self.history = {
            "epoch": {
                "train": {"loss": {}, "metrics": {}},
                "train_eval": {"loss": {}, "metrics": {}},
                "dev": {"loss": {}, "metrics": {}},
                "test": {"loss": {}, "metrics": {}},
            },
            "step": {
                "train": {"loss": {}, "metrics": {}},
                "train_eval": {"loss": {}, "metrics": {}},
                "dev": {"loss": {}, "metrics": {}},
                "test": {"loss": {}, "metrics": {}},
            },
            "no_climbing_epoch_cnt": 0,
            "no_climbing_step_cnt": 0,
            "best_metric": -100.0,
            "best_loss": float("inf"),
            "best_epoch": -1,
            "best_step": -1,
            "curr_epoch": 0,
            "curr_batch": 0,
            "total_steps": 0,
            "current_train_loss": 0.0,
        }

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

    def reset_history(self, reset_all: Optional[bool] = False):
        self.history["no_climbing_epoch_cnt"] = 0
        self.history["no_climbing_step_cnt"] = 0
        self.history["best_metric"] = -100.0
        self.history["best_loss"] = float("inf")
        self.history["best_epoch"] = -1
        self.history["best_step"] = -1
        self.history["curr_epoch"] = 0
        self.history["curr_batch"] = 0
        self.history["total_steps"] = 0
        self.history["current_train_loss"] = 0.0

        if reset_all:
            for type_key in ["epoch", "step"]:
                for data_type in ["train", "train_eval", "dev", "test"]:
                    for measure_type in ["loss", "metrics"]:
                        self.history[type_key][data_type][measure_type].clear()

    def load(
        self,
        path: str,
        load_config: Optional[bool] = False,
        load_model: Optional[bool] = True,
        load_optimizer: Optional[bool] = False,
        load_history: Optional[bool] = True,
    ):
        if os.path.exists(path):
            logger.info("Resume checkpoint from {}".format(path))
        else:
            raise ValueError("Checkpoint does not exist, {}".format(path))

        if torch.cuda.device_count() == 0:
            logger.debug("Load store_dict into cpu")
            store_dict = torch.load(path, map_location="cpu")
        else:
            logger.debug(f"Load store_dict into {accelerator.device}")
            store_dict = torch.load(path, map_location=accelerator.device)

        if load_config:
            self.config = OmegaConf.load(
                os.path.join(self.config.task_dir, CONFIG_PARAMS_FILENAME)
            )

        if load_model:
            if self.model and "model_state" in store_dict:
                unwrapped_model = accelerator.unwrap_model(self.model)
                unwrapped_model.load_state_dict(store_dict["model_state"])
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

        if load_history:
            history = store_dict.pop("history")
            self.reset_history(reset_all=True)
            if history is not None:
                self.history = history
            else:
                logger.info(
                    "Loaded history is None, reset history to empty.", level="WARNING"
                )

    def load_best_ckpt(
        self,
        load_optimizer: Optional[bool] = False,
        load_config: Optional[bool] = False,
        load_history: Optional[bool] = True,
    ):
        _task_dir = Path(self.config.task_dir)
        model_name = self.model.__class__.__name__
        ckpt_filepath = str(
            _task_dir.joinpath(
                CHECKPOINT_DIRNAME,
                BEST_CHECKPOINT_FILENAME_TEMPLATE.format(model_name),
            )
        )
        logger.info(f"Loading model from: {ckpt_filepath}")
        self.load(
            ckpt_filepath,
            load_model=True,
            load_optimizer=load_optimizer,
            load_config=load_config,
            load_history=load_history,
        )

    @rank_zero_only
    def save(self, path):
        logger.info(f"Dumping checkpoint into: {path}")
        store_dict = {}

        if self.model:
            model = accelerator.unwrap_model(self.model)
            model_state = model.state_dict()
            store_dict["model_state"] = model_state
        else:
            logger.info("No model state is dumped", logging.WARNING)

        if self.optimizer:
            store_dict["optimizer_state"] = self.optimizer.state_dict()
        else:
            logger.info("No optimizer state is dumped", logging.WARNING)

        store_dict["history"] = self.history

        torch.save(store_dict, path)

    def save_ckpt(self, identifier: Optional[str] = BEST_IDENTIFIER):
        ckpt_dir = os.path.join(self.config.task_dir, CHECKPOINT_DIRNAME)
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        ckpt_name = CHECKPOINT_FILENAME_TEMPLATE.format(
            self.model.__class__.__name__, identifier
        )
        self.save(os.path.join(ckpt_dir, ckpt_name))

    @classmethod
    def from_config(
        cls,
        config: OmegaConf,
        update_config: Optional[dict] = None,
        **kwargs,
    ):
        cls.logging(f"Initializing from configuration: {OmegaConf.to_yaml(config)}")

        # update
        if update_config is not None:
            for key, val in update_config.items():
                setattr(config, key, val)

        kwargs["initialize"] = kwargs.pop("initialize", True)
        kwargs["makedirs"] = kwargs.pop("makedirs", True)
        kwargs["dump_configfile"] = kwargs.pop("dump_configfile", True)
        return cls(config, **kwargs)

    @classmethod
    def from_configfile(
        cls,
        config_filepath: str,
        **kwargs,
    ):
        cls.logging(f"Initializing from configuration file: {config_filepath}")
        config = OmegaConf.load(config_filepath)

        return cls.from_config(config, **kwargs)

    @classmethod
    def from_taskdir(
        cls,
        task_dir: str,
        load_best_model: Optional[bool] = True,
        load_best_optimizer: Optional[bool] = False,
        load_config: Optional[bool] = False,
        **kwargs,
    ):
        _task_dir = Path(task_dir)
        config_filepath = str(_task_dir.joinpath(CONFIG_PARAMS_FILENAME).absolute())
        ins = cls.from_configfile(config_filepath, **kwargs)
        if load_best_model:
            ins.load_best_ckpt(
                load_optimizer=load_best_optimizer, load_config=load_config
            )
        return ins
