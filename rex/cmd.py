import os
import sys
import shutil
from pathlib import Path

import rex
from rex.tasks.task import Task
from rex.utils.build_emb import build_emb
from rex.utils.config import ConfigParser, ConfigArgument
from rex.utils.registry import get_registered, register, NAMESPACE_REGISTRY
from rex.utils.logging import logger


REX_DIR = Path(rex.__file__).parent
TEMPLATES_DIR = REX_DIR.joinpath("templates")


@register("rex_init_call")
def new():
    new_args = ConfigParser.parse_cmd(
        ConfigArgument(
            "task_type",
            choices=["classification", "tagging"],
            help="Type of the new task",
        ),
        ConfigArgument(
            "task_name", help="Name of the task (used for creating new task folder)"
        ),
        init_priority_args=False,
        cmd_args=sys.argv[2:],
    )
    task_dir = Path(new_args.task_name)
    task_dir.mkdir(parents=True, exist_ok=False)
    src_dir = TEMPLATES_DIR.joinpath(new_args.task_type)
    for filename in os.listdir(src_dir):
        src_fp = src_dir.joinpath(filename)
        dest_fp = task_dir.joinpath(filename)
        shutil.copy2(src_fp, dest_fp)
    print(f"New {new_args.task_type} task in {task_dir.absolute()}")


@register("rex_init_call")
def train():
    config = ConfigParser.parse_cmd(cmd_args=sys.argv[2:])
    if "task_type" not in config:
        raise ValueError(
            (
                "`task_type` must exist in config! "
                "If you are using the default base config, `task_type` is not initialised in it."
            )
        )
    TASK_CLASS = get_registered("task", config.task_type)
    task: Task = TASK_CLASS(config)
    logger.info("Task initialised, ready to start")
    task.train()


@register("rex_init_call")
def emb():
    config = ConfigParser.parse_cmd(
        ConfigArgument("-r", "--raw-emb-filepath", help="raw embedding filepath"),
        ConfigArgument("-o", "--dump-emb-filepath", help="dumped embedding filepath"),
        ConfigArgument("-f", "--filepaths", nargs="*", help="raw text filepaths"),
        cmd_args=sys.argv[2:],
    )
    build_emb(config.raw_emb_filepath, config.dump_emb_filepath, *config.filepaths)


def main():
    args = ConfigParser.parse_cmd(
        ConfigArgument("command", choices=["new", "train", "emb"], help="REx mode."),
        init_priority_args=False,
        cmd_args=sys.argv[1:2],
    )
    logger.info(f"Entering {args.command} mode...")

    if args.command in NAMESPACE_REGISTRY["rex_init_call"]:
        command = get_registered("rex_init_call", args.command)
        command()
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
