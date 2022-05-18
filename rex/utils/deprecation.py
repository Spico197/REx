from rex.utils.logging import logger


def deprecation_warning(deprecated: str, new: str):
    logger.warning(f"`{deprecated}` is deprecated, use `{new}` instead.")
