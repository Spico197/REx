# ref: https://github.com/bytedance/ParaGen
import functools
from typing import Callable
from collections import defaultdict


CLASS_REGISTRY = {}


def register_class(cls):
    """
    Register a class with its name
    Args:
        cls: a new class fro registration
    """
    name = cls.__name__
    if name in CLASS_REGISTRY:
        raise ValueError(f"Cannot register duplicate `CLASS_REGISTRY` class ({name})")
    CLASS_REGISTRY[name] = cls
    return cls


def get_registered_class(name):
    if name not in CLASS_REGISTRY:
        raise ValueError(f"{name} not registered!")
    return CLASS_REGISTRY[name]


NAMESPACE_REGISTRY = defaultdict(dict)


def register(namespace: str):
    def register_on_namespace(call: Callable):
        cname = call.__name__
        if cname in NAMESPACE_REGISTRY[namespace]:
            raise ValueError(
                f"Cannot register duplicate {cname} in `NAMESPACE_REGISTRY[{namespace}]`"
            )
        NAMESPACE_REGISTRY[namespace][cname] = call
        return call

    return register_on_namespace


def get_registered(namespace: str, call_name: str):
    if call_name not in NAMESPACE_REGISTRY[namespace]:
        raise ValueError(f"{call_name} not registered in {namespace}!")
    call = NAMESPACE_REGISTRY[namespace][call_name]
    return call
