import argparse
import importlib
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Iterable, List, Optional

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
            "-d",
            "--use-default-base-config",
            default=False,
            action="store_true",
            help="whether to use internal base config",
        )
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
    def parse_cmd_args(cls, *args, cmd_args: Optional[List[str]] = None, **kwargs):
        parser = cls(*args, **kwargs)
        args = parser.parse_args(args=cmd_args)
        return parser, args

    @classmethod
    def parse_args_config(cls, parser, args):
        if parser._inited:
            for path in ["base_config_filepath", "custom_config_filepath"]:
                # arg: pathlib.Path
                arg = getattr(args, path)
                if arg is not None:
                    assert arg.exists() is True and arg.is_file() is True
                    setattr(args, path, str(arg.absolute()))

        config = parser.convert_args_to_config(args)

        return config

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
            $ python run.py -d
            $ python run.py -b conf/config.yaml
            $ python run.py -c conf/re/sent_ipre.yaml
            $ python run.py -a lr.encoder=1e-4 dropout=0.6
            $ python run.py -b conf/config.yaml -c conf/re/sent_ipre.yaml
            $ python run.py -b conf/config.yaml -a lr.encoder=1e-4 dropout=0.6
            $ python run.py -c conf/re/sent_ipre.yaml -a lr.encoder=1e-4 dropout=0.6
            $ python run.py -b conf/config.yaml -c conf/re/sent_ipre.yaml -a lr.encoder=1e-4 dropout=0.6
        """
        parser, args = cls.parse_cmd_args(*args, cmd_args=cmd_args, **kwargs)
        config = cls.parse_args_config(parser, args)

        return config

    def convert_args_to_config(self, args: argparse.Namespace) -> OmegaConf:
        """Convert args into config"""
        config = OmegaConf.create(
            {"_config_info": {"create_time": datetime.now().strftime("%F %X")}}
        )
        OmegaConf.set_readonly(config, False)

        if "use_default_base_config" in args:
            config._config_info.use_default_base_config = args.use_default_base_config
            if args.use_default_base_config:
                default_config = DefaultBaseConfig()
                config.merge_with(asdict(default_config))

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


@dataclass
class DefaultBaseConfig:
    # task
    task_name: str = field(
        default="temp_task",
        metadata={"help": "name of task, used for creating task directory"},
    )
    # filepaths
    output_dir: str = field(
        default="outputs", metadata={"help": "base output directory"}
    )
    task_dir: str = field(
        default="outputs/temp_task",
        metadata={"help": "task directory"},
    )
    data_dir: str = field(default="data", metadata={"help": "data directory"})
    train_filepath: str = field(
        default="train.jsonl", metadata={"help": "filepath to train set"}
    )
    dev_filepath: str = field(
        default="dev.jsonl", metadata={"help": "filepath to dev set"}
    )
    test_filepath: str = field(
        default="test.jsonl", metadata={"help": "filepath to test set"}
    )
    # training control
    device: str = field(default="cpu", metadata={"help": "device string"})
    random_seed: int = field(default=1227, metadata={"help": "random seed"})
    num_epochs: int = field(
        default=50, metadata={"help": "max number of training epochs"}
    )
    num_steps: int = field(
        default=-1,
        metadata={
            "help": "max number of training steps. `-1` if not training in steps"
        },
    )
    epoch_patience: int = field(
        default=5,
        metadata={"help": "stop training if the metric does not grow in [x] epochs"},
    )
    step_patience: int = field(
        default=5000,
        metadata={"help": "stop training if the metric does not grow in [x] steps"},
    )
    batch_size: int = field(
        default=64,
        metadata={
            "help": "training batch size for each process (if ddp, the real batch size is num_process * batch_size)"
        },
    )
    learning_rate: float = field(
        default=1e-3, metadata={"help": "training learning rate"}
    )
    max_grad_norm: float = field(
        default=-1.0, metadata={"help": "max gradient norm. `-1`: no clipping"}
    )
    skip_train: bool = field(
        default=False, metadata={"help": "whether to skip training process"}
    )
    debug_mode: bool = field(
        default=False,
        metadata={
            "help": "whether to enter debug mode (use less data for debug and test)"
        },
    )
    grad_accum_steps: int = field(
        default=1, metadata={"help": "gradient accumulation steps"}
    )
    resumed_training_path: str = field(
        default=None,
        metadata={
            "help": "path to load checkpoint for resumed training, `None` if not resumed training"
        },
    )
    step_eval_interval: int = field(
        default=-1, metadata={"help": "evaluation interval steps. `-1` if not use"}
    )
    epoch_eval_interval: int = field(
        default=1, metadata={"help": "evaluation interval epochs. `-1` if not use"}
    )
    eval_on_data: list = field(
        default_factory=lambda: ["dev"],
        metadata={"help": "evaluation on which data set(s)"},
    )
    select_best_on_data: str = field(
        default="dev", metadata={"help": "use which data to select best model"}
    )
    select_best_by_key: str = field(
        default="metric",
        metadata={"help": "the best model selection strategy, `metric` or `loss`"},
    )
    best_metric_field: str = field(
        default="micro.f1",
        metadata={
            "help": "metric value index to a dict (use `.` for solving nested dict)"
        },
    )
    save_every_ckpt: bool = field(
        default=False,
        metadata={"help": "whether to save every checkpoint on evaluation interval"},
    )
    save_best_ckpt: bool = field(
        default=True, metadata={"help": "whether to save the best checkpoint"}
    )
    final_eval_on_test: bool = field(
        default=True, metadata={"help": "whether to evaluate on final test set"}
    )
    # misc
    main_process_logging: bool = field(
        default=True, metadata={"help": "whether to logging in only master node"}
    )


if __name__ == "__main__":
    config = DefaultBaseConfig()
    print(config)
    print(asdict(config))
