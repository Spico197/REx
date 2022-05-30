import datetime
import functools
from typing import Callable, Optional

from accelerate.state import AcceleratorState

from rex.utils.logging import logger


def safe_try(
    func_placeholder: Optional[Callable] = None,
    start_msg: Optional[str] = None,
    end_msg: Optional[str] = None,
):
    def func_wrapper(func):
        @functools.wraps(func)
        def try_func(*args, **kwargs):
            if start_msg is not None:
                logger.info(str(start_msg))

            err_happened = False
            start_dt = datetime.datetime.now()

            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                err_happened = True
                raise KeyboardInterrupt
            except Exception as err:
                err_happened = True
                logger.exception(err)
            finally:
                time_delta = datetime.datetime.now() - start_dt
                if not err_happened:
                    logger.debug(f"Func `{func.__name__}` finished without err.")
                if end_msg is not None:
                    logger.info(str(end_msg))
                logger.info(f"Func: `{func.__name__}` call time: {str(time_delta)}")

        return try_func

    if func_placeholder is None:
        return func_wrapper
    else:
        return func_wrapper(func_placeholder)


def rank_zero_only(fn):
    # inspired by pytorch-lightning / rank_zero
    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if AcceleratorState().local_process_index == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn
