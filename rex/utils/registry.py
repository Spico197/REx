# ref: https://github.com/bytedance/ParaGen
import importlib
import os
import pkgutil
import sys
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional, Set

CLASS_REGISTRY = {}


def register_class(cls):
    """
    Register a class with its name
    Args:
        cls: a new class fro registration
    """
    name = cls.__name__
    # if name in CLASS_REGISTRY:
    #     raise ValueError(f"Cannot register duplicate `CLASS_REGISTRY` class ({name})")
    CLASS_REGISTRY[name] = cls
    return cls


def get_registered_class(name):
    # if name not in CLASS_REGISTRY:
    #     raise ValueError(f"{name} not registered!")
    return CLASS_REGISTRY[name]


NAMESPACE_REGISTRY = defaultdict(dict)


def register(namespace: str):
    def register_on_namespace(call: Callable):
        cname = call.__name__
        # if cname in NAMESPACE_REGISTRY[namespace]:
        #     raise ValueError(
        #         f"Cannot register duplicate {cname} in `NAMESPACE_REGISTRY[{namespace}]`"
        #     )
        NAMESPACE_REGISTRY[namespace][cname] = call
        return call

    return register_on_namespace


def get_registered(namespace: str, call_name: str):
    # if call_name not in NAMESPACE_REGISTRY[namespace]:
    #     raise ValueError(f"{call_name} not registered in {namespace}!")
    call = NAMESPACE_REGISTRY[namespace][call_name]
    return call


def _get_module_name(path: str):
    return ".".join(path.replace("/__init__.py", "").split("/")).replace(".py", "")


def _import_module(modules_dir: str) -> None:
    simplified_module_dir = modules_dir[modules_dir.find("rex/") :]
    for file in os.listdir(modules_dir):
        path = os.path.join(modules_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module = ".".join(
                os.path.join(simplified_module_dir, file)
                .replace("/__init__.py", "")
                .split("/")
            ).replace(".py", "")
            importlib.import_module(module)


def call_register(path):
    """
    Args:
        path: __file__
    """
    module_dir = os.path.dirname(path)
    if os.path.exists(module_dir):
        _import_module(module_dir)
    else:
        raise ValueError(f"Dir {module_dir} not exists!")


@contextmanager
def push_python_path(path):
    """
    Prepends the given path to `sys.path`.
    This method is intended to use with `with`, so after its usage, its value willbe removed from
    `sys.path`.
    """
    # In some environments, such as TC, it fails when sys.path contains a relative path, such as ".".
    path = Path(path).resolve()
    path = str(path)
    sys.path.insert(0, path)
    try:
        yield
    finally:
        # Better to remove by value, in case `sys.path` was manipulated in between.
        sys.path.remove(path)


def import_module_and_submodules(
    package_name: str, exclude: Optional[Set[str]] = None
) -> None:
    """
    Import all public submodules under the given package.
    Primarily useful so that people using AllenNLP as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    Idea borrowed from [AllenNLP]_.

    References:
        .. [AllenNLP] https://github.com/allenai/allennlp/blob/068407e3d476750ee75fd52840be7a160b693760/allennlp/common/util.py#L266
    """
    # take care of None
    exclude = exclude if exclude else set()
    if package_name in exclude:
        return

    importlib.invalidate_caches()

    # For some reason, python doesn't always add this by default to your path, but you pretty much
    # always want it when using `--include-package`.  And if it's already there, adding it again at
    # the end won't hurt anything.
    with push_python_path("."):
        # Import at top level
        module = importlib.import_module(package_name)
        path = getattr(module, "__path__", [])
        path_string = "" if not path else path[0]

        # walk_packages only finds immediate children, so need to recurse.
        for module_finder, name, _ in pkgutil.walk_packages(path):
            # Sometimes when you import third-party libraries that are on your path,
            # `pkgutil.walk_packages` returns those too, so we need to skip them.
            if path_string and module_finder.path != path_string:  # type: ignore[union-attr]
                continue
            if name.startswith("_"):
                # skip directly importing private subpackages
                continue
            if name.startswith("test"):
                continue
            subpackage = f"{package_name}.{name}"
            import_module_and_submodules(subpackage, exclude=exclude)


def import_submodules(package_name: str) -> None:
    """
    Import all submodules under the given package.
    Primarily useful so that people using REx as a library
    can specify their own custom packages and have their custom
    classes get loaded and registered.
    Idea borrowed from [AllenNLP]_.

    References:
        .. [AllenNLP] https://github.com/allenai/allennlp/blob/068407e3d476750ee75fd52840be7a160b693760/allennlp/common/util.py#L266
    """
    importlib.invalidate_caches()

    # Import at top level
    module = importlib.import_module(package_name)
    path = getattr(module, "__path__", "")

    # walk_packages only finds immediate children, so need to recurse.
    for _, name, _ in pkgutil.walk_packages(path):
        subpackage = f"{package_name}.{name}"
        import_submodules(subpackage)
