__version__ = "0.1.4"

__all__ = ["__version__", "data", "metrics", "models", "modules", "tasks", "utils"]


from accelerate import Accelerator

accelerator = Accelerator()

from .utils.registry import call_register
