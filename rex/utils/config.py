import os
import yaml
import json
import argparse
import pathlib
from datetime import datetime
from typing import Optional, List

from omegaconf import OmegaConf

from rex.utils.logging import logger
from rex.utils.initialization import set_seed_and_log_path


def get_cmd_arg_parser() -> argparse.ArgumentParser:
    """Get ArgumentParser"""
    arg_parser = argparse.ArgumentParser(
        description="""Get config (~OmegaConf) from terminal commands"""
    )
    arg_parser.add_argument(
        "-b",
        "--base-config-filepath",
        type=pathlib.Path,
        help="base configuration filepath, could be a template, lowest priority",
    )
    arg_parser.add_argument(
        "-c",
        "--custom-config-filepath",
        type=pathlib.Path,
        help="configuration filepath, customised configs, middle priority",
    )
    arg_parser.add_argument(
        "-a",
        "--additional-args",
        type=str,
        nargs="*",
        help="additional args in dot-list format, highest priority",
    )
    return arg_parser


def get_config_from_cmd(cmd_args: Optional[List[str]] = None) -> OmegaConf:
    """
    Get command arguments from `base_config_filepath`, `custom_config_filepath` and `additional_args`.

    To make configurations flexible, configs are split into three levels:
        - `base_config_filepath`: path to basic configuration file in yaml format, lowest priority
        - `custom_config_filepath`: path to customised config file in yaml format, middle priority.
            If this file is loaded, it will override configs in the `base_config_filepath`.
        - `additional_args`: additional args in dot-list format, highest priority.
            If these args are set, it will override configs in the previous files.

    Args:
        cmd_args: if you don't want to use the command line, provide `cmd_args` like `sys.argv[1:]`

    Examples:

        $ python run.py -b conf/config.yaml
        $ python run.py -c conf/re/sent_ipre.yaml
        $ python run.py -a lr.encoder=1e-4 dropout=0.6
        $ python run.py -b conf/config.yaml -c conf/re/sent_ipre.yaml
        $ python run.py -b conf/config.yaml -a lr.encoder=1e-4 dropout=0.6
        $ python run.py -c conf/re/sent_ipre.yaml -a lr.encoder=1e-4 dropout=0.6
        $ python run.py -b conf/config.yaml -c conf/re/sent_ipre.yaml -a lr.encoder=1e-4 dropout=0.6
    """
    arg_parser = get_cmd_arg_parser()
    args = arg_parser.parse_args(args=cmd_args)

    for path in ["base_config_filepath", "custom_config_filepath"]:
        # arg: pathlib.Path
        arg = getattr(args, path)
        if arg is not None:
            assert arg.exists() is True and arg.is_file() is True
            setattr(args, path, str(arg.absolute()))

    config = convert_args_to_config(args)

    return config


def convert_args_to_config(args: argparse.Namespace) -> OmegaConf:
    """Convert args into config"""
    config = OmegaConf.create(
        {"_config_info": {"create_time": datetime.now().strftime("%F %X")}}
    )
    OmegaConf.set_readonly(config, False)

    base_config_filepath = args.base_config_filepath
    if base_config_filepath is not None:
        base_config = OmegaConf.load(base_config_filepath)
        config.merge_with(base_config)
        config._config_info.base_filepath = base_config_filepath

    custom_config_filepath = args.custom_config_filepath
    if custom_config_filepath is not None:
        custom_config = OmegaConf.load(custom_config_filepath)
        config.merge_with(custom_config)
        config._config_info.custom_config_filepath = custom_config_filepath

    additional_args = args.additional_args
    if additional_args is not None:
        add_args = OmegaConf.from_dotlist(additional_args)
        config.merge_with(add_args)
        config._config_info.additional_args = additional_args

    return config


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
