from rex.utils.build_emb import build_emb
from rex.utils.config import ConfigArgument, ConfigParser
from rex.utils.registry import register


@register("rex_init_call")
def emb(cmd_args=None):
    config = ConfigParser.parse_cmd(
        ConfigArgument("-r", "--raw-emb-filepath", help="raw embedding filepath"),
        ConfigArgument("-o", "--dump-emb-filepath", help="dumped embedding filepath"),
        ConfigArgument("-f", "--filepaths", nargs="*", help="raw text filepaths"),
        cmd_args=cmd_args,
    )
    build_emb(config.raw_emb_filepath, config.dump_emb_filepath, *config.filepaths)


if __name__ == "__main__":
    emb()
