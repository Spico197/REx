__version__ = "0.1.4"

__all__ = ["__version__", "data", "metrics", "models", "modules", "tasks", "utils"]


from accelerate import Accelerator, DistributedDataParallelKwargs

ddp_args = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_args])

from .utils.registry import call_register
