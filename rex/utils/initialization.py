import os
import random
import sys
from typing import Optional

import numpy as np
import torch
from loguru import logger


def set_seed(seed: Optional[int] = 1227, set_cudnn: Optional[bool] = False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if set_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def set_seed_and_log_path(
    seed: Optional[int] = 1227,
    log_path: Optional[str] = None,
    set_cudnn: Optional[bool] = False
):
    set_seed(seed, set_cudnn=set_cudnn)
    if log_path:
        logger.add(log_path, backtrace=True, diagnose=True)


def set_device(device):
    torch.cuda.set_device(device)
