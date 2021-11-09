import os
import yaml
import json
import argparse
from typing import Optional

from loguru import logger

from rex.utils.initialization import set_seed_and_log_path


class ArgConfig(object):
    def __init__(self, **kwargs):
        parser = argparse.ArgumentParser()
        parser.add_argument("-c", "--config", type=str, help="YAML config filepath")
        parser.add_argument(
            "--local_rank",
            required=False,
            default=-1,
            type=int,
            help="remained for distributed" " data parallel support",
        )
        args = parser.parse_args()
        self.local_rank = args.local_rank
        if args.config:
            params = YamlConfig(args.config, **kwargs)
            for key, val in params.__dict__.items():
                setattr(self, key, val)


class YamlConfig(object):
    def __init__(self, config_filepath: str, make_taskdir: Optional[bool] = True):
        self.config_abs_path = os.path.abspath(config_filepath)

        # do not use `with` expression here in order to find errors ealier
        fin = open(self.config_abs_path, "rt", encoding="utf-8")
        __config = yaml.safe_load(fin)
        fin.close()

        for key, val in __config.items():
            setattr(self, key, val)

        if make_taskdir:
            self.output_dir = os.path.join(self.output_dir, self.task_name)
        __config.update({"output_dir": self.output_dir})
        self._config = __config

        overwritten_flag = False
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        else:
            overwritten_flag = True

        set_seed_and_log_path(
            self.random_seed, os.path.join(self.output_dir, "log.log")
        )

        if overwritten_flag:
            logger.warning("Overwrite output directory.")

        with open(
            os.path.join(self.output_dir, "config.yaml"), "wt", encoding="utf-8"
        ) as fout:
            yaml.dump(__config, fout)

    def __str__(self, indent=0):
        return json.dumps(self.__dict__, indent=indent)
