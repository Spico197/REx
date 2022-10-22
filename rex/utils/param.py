import torch.nn as nn


def calc_module_params(module: nn.Module) -> int:
    """
    Get the number of parameters in a module
    """
    tot_params = 0
    for param in module.parameters():
        tot_params += param.nelement()
    return tot_params
