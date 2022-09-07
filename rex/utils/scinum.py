"""
Scientific number
"""

import math

import numpy as np


def is_close(num1, num2, rel_tol=None, abs_tol=None):
    return math.isclose(num1, num2, rel_tol=rel_tol, abs_tol=abs_tol)


def convert_numpy_obj_to_native(obj):
    if isinstance(obj, dict):
        for key in obj:
            convert_numpy_obj_to_native(obj[key])
