import torch

from rex.utils.tensor_move import move_to_cuda_device
from tests import RunIf


@RunIf(min_gpus=1)
def test_tensor_move():
    device = torch.device("cuda:0")
    x = torch.randn(5, 5)
    x_cu = move_to_cuda_device(x, device)
    assert x_cu.device == device


@RunIf(min_gpus=1)
def test_dict_tensor_move():
    device = torch.device("cuda:0")
    x = {
        "tensor": torch.randn(5, 5),
        "list": [1, 2, 3],
        "recursive": {"tensor": torch.randn(5, 5), "nums": [1, 2, 3]},
    }
    x_cu = move_to_cuda_device(x, device)
    assert x_cu["tensor"].device == device
    assert x_cu["recursive"]["tensor"].device == device
