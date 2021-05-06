from functools import partial

from tqdm import tqdm


tqdm = partial(tqdm, ncols=80, ascii=True)
