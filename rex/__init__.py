__all__ = ["__version__", "data", "metrics", "models", "modules", "tasks", "utils"]

from accelerate import Accelerator, DistributedDataParallelKwargs

from rex.version import __version__

ddp_args = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[ddp_args])
