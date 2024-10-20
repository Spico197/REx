import os

import torch.distributed as dist


def get_world_size(group: dist.ProcessGroup):
    if dist.is_initialized():
        return dist.get_world_size(group)
    else:
        return -1


def get_local_world_size():
    return int(os.environ.get("LOCAL_WORLD_SIZE", -1))


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", -1))


def is_local_rank0():
    return get_local_rank() in [0, -1]


def get_global_rank(group: dist.ProcessGroup = None):
    if dist.is_initialized():
        return dist.get_rank(group)
    else:
        return -1


def is_global_rank0(group: dist.ProcessGroup = None):
    return get_global_rank(group) in [0, -1]
