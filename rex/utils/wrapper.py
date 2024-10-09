"""
Some decorators are imported from:
    https://towardsdatascience.com/python-decorators-for-data-science-6913f717669a
"""

import datetime
import functools
import smtplib
import time
import traceback
from email.mime.text import MIMEText
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


def retry(max_tries=3, delay_seconds=1):
    def decorator_retry(func):
        @functools.wraps(func)
        def wrapper_retry(*args, **kwargs):
            tries = 0
            while tries < max_tries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    tries += 1
                    if tries == max_tries:
                        raise e
                    time.sleep(delay_seconds)

        return wrapper_retry

    return decorator_retry


def memoize(func):
    cache = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args in cache:
            return cache[args]
        else:
            result = func(*args)
            cache[args] = result
            return result

    return wrapper


def timing(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to run.")
        return result

    return wrapper


def email_on_failure(sender_email, password, recipient_email):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # format the error message and traceback
                err_msg = f"Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"

                # create the email message
                message = MIMEText(err_msg)
                message["Subject"] = f"{func.__name__} failed"
                message["From"] = sender_email
                message["To"] = recipient_email

                # send the email
                with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                    smtp.login(sender_email, password)
                    smtp.sendmail(sender_email, recipient_email, message.as_string())

                # re-raise the exception
                raise

        return wrapper

    return decorator
