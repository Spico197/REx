import sys

from rex.utils.config import ConfigArgument, ConfigParser
from rex.utils.registry import NAMESPACE_REGISTRY, get_registered, register
from rex.version import __version__

from .emb import emb
from .new import new
from .train import train


@register("rex_init_call")
def version(*args, **kwargs):
    print(f"{__version__}")


def main():
    args = ConfigParser.parse_cmd(
        ConfigArgument(
            "command",
            choices=list(NAMESPACE_REGISTRY["rex_init_call"].keys()),
            help="REx mode.",
        ),
        init_priority_args=False,
        cmd_args=sys.argv[1:2],
        description=f"REx (version: {__version__}) - A toolkit for Relation, Event eXtraction (REx) and more...",
    )

    if args.command in NAMESPACE_REGISTRY["rex_init_call"]:
        command = get_registered("rex_init_call", args.command)
        command(cmd_args=sys.argv[2:])
    else:
        print(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
