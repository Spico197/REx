from rex.io.vocab import Vocabulary
from rex.utils.logger import Logger
from rex.utils.config import ArgConfig


CONFIG = ArgConfig()
logger = Logger(CONFIG.output_dir)


def train():
    raise NotImplementedError


def test():
    raise NotImplementedError


if __name__ == "__main__":
    train()
    test()
