import os
import sys
import logging
from typing import Optional


class Logger(object):
    def __init__(self, output_dir: str,
                 name: Optional[str] = "rex",
                 level: Optional[int] = logging.DEBUG,
                 log_filename: Optional[str] = "log.log"):
        self._logger = logging.getLogger(name)
        fmt = ("[%(asctime)-15s]-%(levelname)s-%(filename)s"
               "-%(lineno)d-%(process)d: %(message)s")
        datefmt = "%a %d %b %Y %H:%M:%S"
        formatter = logging.Formatter(fmt, datefmt)
        self._logger.setLevel(level)
        log_path = os.path.join(output_dir, log_filename)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        # to avoid print many times
        if not self._logger.handlers:
            self._logger.addHandler(file_handler)
            self._logger.addHandler(stream_handler)
        self._logger.propagate = False
        sys.excepthook = self.handle_exception
        if overwrite:
            self._logger.warn("Output Dir Already Exists, Overwriting")
        
        self.debug = self._logger.debug
        self.info = self._logger.info
        self.warn = self._logger.warn
        self.warning = self._logger.warning
        self.error = self._logger.error

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        self._logger.error("Uncaught exception",
                           exc_info=(exc_type, exc_value, exc_traceback))
