import random
import string
from typing import Optional

from rex.utils import get_now_string


def generate_random_string(length: Optional[int] = 8) -> str:
    return "".join(random.sample(string.ascii_letters + string.digits, length))


def generate_random_string_with_datetime(time_mode: str = "timestamp", random_length: int = 8):
    timestamp = get_now_string(time_mode)
    string = generate_random_string(random_length)
    return f"{timestamp}-{string}"
