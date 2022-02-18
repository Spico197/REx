import os
import sys
import shutil
from pathlib import Path

import rex
from rex.utils.config import ConfigParser, ConfigArgument


REX_DIR = Path(rex.__file__).parent
TEMPLATES_DIR = REX_DIR.joinpath("templates")


def main():
    args = ConfigParser.parse_cmd(
        ConfigArgument("command", choices=["new"], help="REx mode."),
        init_priority_args=False,
        cmd_args=sys.argv[1:2],
    )
    if args.command == "new":
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
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)
