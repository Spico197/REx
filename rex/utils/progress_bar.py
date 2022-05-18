from tqdm import tqdm as _tqdm

from rex.utils.deprecation import deprecation_warning


class pbar(_tqdm):
    def __init__(self, *args, **kwargs):
        if "ncols" not in kwargs:
            kwargs["ncols"] = 80
        if "ascii" not in kwargs:
            kwargs["ascii"] = True
        super().__init__(*args, **kwargs)


class rbar(pbar):
    def __init__(self, iterator, *args, **kwargs):
        iterator = range(iterator, *args)
        super().__init__(iterator, **kwargs)


class tqdm(pbar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        deprecation_warning(self.__class__.__name__, pbar.__name__)
