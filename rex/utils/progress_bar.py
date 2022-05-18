from tqdm import tqdm as _tqdm


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
