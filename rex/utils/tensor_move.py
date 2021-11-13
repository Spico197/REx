from typing import Optional
import torch


def move_to_cuda_device(obj: dict, device: Optional[torch.device] = None):
    for key in obj:
        if isinstance(obj[key], torch.Tensor):
            obj[key] = obj[key].cuda(device)
        elif isinstance(obj[key], dict):
            move_to_cuda_device(obj[key], device)
    return obj
