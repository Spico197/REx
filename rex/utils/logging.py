from accelerate.state import AcceleratorState
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from tqdm import tqdm as _tqdm


class MultiProcessLogger(_Logger):
    def __init__(self, *args, main_process_logging=True):
        super().__init__(*args)

        self.main_process_logging = main_process_logging

    @staticmethod
    def _should_log(main_process_only):
        "Check if log should be performed"
        return not main_process_only or (
            main_process_only and AcceleratorState().local_process_index == 0
        )

    def _log(self, *args):
        """
        Delegates logger call after checking if we should log.

        Accepts a new kwarg of `main_process_only`, which will dictate whether it will be logged across all processes
        or only the main executed one. Default is `True` if not passed
        """
        if self._should_log(self.main_process_logging):
            super()._log(*args)


logger = MultiProcessLogger(_Core(), None, 0, False, False, True, False, True, None, {})
logger.add(
    lambda msg: _tqdm.write(msg, end=""), colorize=True, backtrace=True, diagnose=True
)
