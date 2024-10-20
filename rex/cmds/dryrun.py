import omegaconf

from rex.tasks.base_task import TaskBase
from rex.utils.config import ConfigArgument, ConfigParser
from rex.utils.logging import logger
from rex.utils.registry import get_registered, import_module_and_submodules, register


@register("rex_init_call")
def dryrun(cmd_args=None):
    parser, args = ConfigParser.parse_cmd_args(
        ConfigArgument(
            "-m",
            "--include-package",
            action="append",
            type=str,
            default=[],
            help="packages to load",
        ),
        cmd_args=cmd_args,
    )
    for package_name in getattr(args, "include_package", []):
        import_module_and_submodules(package_name)
    config = ConfigParser.parse_args_config(parser, args)
    config_yaml = omegaconf.OmegaConf.to_yaml(config, resolve=True)
    print(config_yaml)


if __name__ == "__main__":
    dryrun()