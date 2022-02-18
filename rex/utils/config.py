import argparse
import pathlib
from datetime import datetime
from typing import Iterable, Optional, List

from omegaconf import OmegaConf


class ConfigArgument(object):
    """Instance for storing config argument"""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


class ConfigParser(argparse.ArgumentParser):
    """Configuration Parser

    Args:
        *args (~Iterable[ConfigArgument]): arguments to be added in ```add_argument``` function
        **kwargs: arguments of ```argparse.ArgumentParser```

    Examples:
        >>> config = ConfigParser.parse_cmd()
        >>> config = ConfigParser(
        ...     ConfigArgument('-d', '--task-dir', type=str, help='path to task directory'),
        ...     ConfigArgument('-n', '--task-name', type=str, help='name of task')
        ... ).parse_cmd()
    """

    def __init__(
        self,
        *args: Iterable[ConfigArgument],
        init_priority_args: Optional[bool] = True,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        for arg in args:
            assert isinstance(arg, ConfigArgument)
            self.add_argument(*arg.args, **arg.kwargs)

        self._inited = False
        if init_priority_args:
            self._inited = True
            self.init_priority_args()

    def init_priority_args(self):
        self.add_argument(
            "-b",
            "--base-config-filepath",
            type=pathlib.Path,
            help="base configuration filepath, could be a template, lowest priority",
        )
        self.add_argument(
            "-c",
            "--custom-config-filepath",
            type=pathlib.Path,
            help="configuration filepath, customised configs, middle priority",
        )
        self.add_argument(
            "-a",
            "--additional-args",
            type=str,
            nargs="*",
            help="additional args in dot-list format, highest priority",
        )

    @classmethod
    def parse_cmd(
        cls, *args, cmd_args: Optional[List[str]] = None, **kwargs
    ) -> OmegaConf:
        """
        Get command arguments from `base_config_filepath`, `custom_config_filepath` and `additional_args`.

        To make configurations flexible, configs are split into three levels:
            - `base_config_filepath`: path to basic configuration file in yaml format, lowest priority
            - `custom_config_filepath`: path to customised config file in yaml format, middle priority.
                If this file is loaded, it will override configs in the `base_config_filepath`.
            - `additional_args`: additional args in dot-list format, highest priority.
                If these args are set, it will override configs in the previous files.

        Args:
            *args: for initialise the config parser object
            cmd_args: if you don't want to use the command line, provide `cmd_args` like `sys.argv[1:]`
            **kwargs: for initialise the config parser object

        Examples:
            $ python run.py -b conf/config.yaml
            $ python run.py -c conf/re/sent_ipre.yaml
            $ python run.py -a lr.encoder=1e-4 dropout=0.6
            $ python run.py -b conf/config.yaml -c conf/re/sent_ipre.yaml
            $ python run.py -b conf/config.yaml -a lr.encoder=1e-4 dropout=0.6
            $ python run.py -c conf/re/sent_ipre.yaml -a lr.encoder=1e-4 dropout=0.6
            $ python run.py -b conf/config.yaml -c conf/re/sent_ipre.yaml -a lr.encoder=1e-4 dropout=0.6
        """
        parser = cls(*args, **kwargs)
        args = parser.parse_args(args=cmd_args)

        if parser._inited:
            for path in ["base_config_filepath", "custom_config_filepath"]:
                # arg: pathlib.Path
                arg = getattr(args, path)
                if arg is not None:
                    assert arg.exists() is True and arg.is_file() is True
                    setattr(args, path, str(arg.absolute()))

        config = parser.convert_args_to_config(args)

        return config

    def convert_args_to_config(self, args: argparse.Namespace) -> OmegaConf:
        """Convert args into config"""
        config = OmegaConf.create(
            {"_config_info": {"create_time": datetime.now().strftime("%F %X")}}
        )
        OmegaConf.set_readonly(config, False)

        if self._inited:
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

        kwargs = dict(args._get_kwargs())
        for k in ["base_config_filepath", "custom_config_filepath", "additional_args"]:
            if k in kwargs:
                kwargs.pop(k)
        config.merge_with(kwargs)

        return config
