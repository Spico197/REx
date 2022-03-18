from typing import Optional, Union

import torch


def move_to_cuda_device(
    obj: Union[torch.Tensor, dict], device: Optional[torch.device] = None
):
    if isinstance(obj, torch.Tensor):
        obj = obj.cuda(device)
    elif isinstance(obj, dict):
        for key in obj:
            obj[key] = move_to_cuda_device(obj[key], device)
    return obj


def move_to_device(
    obj: Union[torch.Tensor, dict], device: Optional[torch.device] = None
):
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    elif isinstance(obj, dict):
        for key in obj:
            obj[key] = move_to_device(obj[key], device)
    return obj
