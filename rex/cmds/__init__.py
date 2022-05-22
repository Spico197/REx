import sys

from rex.utils.config import ConfigArgument, ConfigParser
from rex.utils.logging import logger
from rex.utils.registry import NAMESPACE_REGISTRY, call_register, get_registered

from .emb import emb
from .new import new
from .train import train


def main():
    args = ConfigParser.parse_cmd(
        ConfigArgument(
            "command",
            choices=list(NAMESPACE_REGISTRY["rex_init_call"].keys()),
            help="REx mode.",
        ),
        init_priority_args=False,
        cmd_args=sys.argv[1:2],
    )
    logger.info(f"Entering {args.command} mode...")

    if args.command in NAMESPACE_REGISTRY["rex_init_call"]:
        command = get_registered("rex_init_call", args.command)
        command(cmd_args=sys.argv[2:])
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
