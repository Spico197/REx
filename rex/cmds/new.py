import os
import shutil
from pathlib import Path

import rex
from rex.utils.config import ConfigArgument, ConfigParser
from rex.utils.registry import register

REX_DIR = Path(rex.__file__).parent
TEMPLATES_DIR = REX_DIR.joinpath("templates")


@register("rex_init_call")
def new(cmd_args=None):
    new_args = ConfigParser.parse_cmd(
        ConfigArgument(
            "task_name", help="Name of the task (used for creating new task folder)"
        ),
        init_priority_args=False,
        cmd_args=cmd_args,
    )
    task_dir = Path(new_args.task_name)
    shutil.copytree(TEMPLATES_DIR, task_dir)
    print(f"New task in {task_dir.absolute()}")


if __name__ == "__main__":
    new()
