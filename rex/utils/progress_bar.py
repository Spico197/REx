from tqdm import tqdm as _tqdm

from rex.utils.distributed import is_global_rank0


class pbar(_tqdm):
    def __init__(self, *args, **kwargs):
        if "ncols" not in kwargs:
            kwargs["ncols"] = 100
        if "ascii" not in kwargs:
            kwargs["ascii"] = True

        main_process_logging = kwargs.pop("main_process_logging", True)
        if main_process_logging and not is_global_rank0():
            kwargs["disable"] = True

        super().__init__(*args, **kwargs)


class rbar(pbar):
    def __init__(self, iterator, *args, **kwargs):
        iterator = range(iterator, *args)
        super().__init__(iterator, **kwargs)
