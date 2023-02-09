import shutil
from pathlib import Path

import rex
from rex.utils.config import ConfigArgument, ConfigParser, DefaultBaseConfig
from rex.utils.registry import register

REX_DIR = Path(rex.__file__).parent
TEMPLATES_DIR = REX_DIR.joinpath("templates")


@register("rex_init_call")
def gen_conf(cmd_args=None):
    new_args = ConfigParser.parse_cmd(
        ConfigArgument(
            "filepath",
            help="filepath to save config file",
            default="config-template.yaml",
        ),
        init_priority_args=False,
        cmd_args=cmd_args,
    )
    DefaultBaseConfig().dump_yaml(new_args.filepath)
    print(f"Generated to {new_args.filepath}")


if __name__ == "__main__":
    gen_conf()
