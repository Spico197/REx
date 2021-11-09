import sys
import logging
from typing import Optional


class Logger(object):
    def __init__(
        self,
        name: str,
        log_path: Optional[str] = None,
        level: Optional[int] = logging.INFO,
    ):
        self.name = name
        self.log_path = log_path

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(level)

        fmt = "[%(asctime)-15s]-%(levelname)s-%(filename)s-%(lineno)d-%(process)d: %(message)s"
        datefmt = "%a %d %b %Y %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        if not self.logger.handlers and log_path is not None:
            fh = logging.FileHandler(log_path)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)

            self.logger.addHandler(fh)
            self.logger.addHandler(stream_handler)

        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error

        self.logger.propagate = False
        sys.excepthook = self.handle_exception

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        self.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
