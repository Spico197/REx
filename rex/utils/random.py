import random
import string
from typing import Optional


def generate_random_string(len: Optional[int] = 8) -> str:
    return "".join(random.sample(string.ascii_letters + string.digits, len))
