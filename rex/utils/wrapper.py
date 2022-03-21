import functools
from typing import Callable, Optional

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

            try:
                return func(*args, **kwargs)
            except KeyboardInterrupt:
                err_happened = True
                raise KeyboardInterrupt
            except Exception as err:
                err_happened = True
                logger.exception(err)
            finally:
                if not err_happened:
                    logger.debug(f"Func {func} finished without err.")
                if end_msg is not None:
                    logger.info(str(end_msg))

        return try_func

    if func_placeholder is None:
        return func_wrapper
    else:
        return func_wrapper(func_placeholder)
