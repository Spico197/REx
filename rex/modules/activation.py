import torch.nn as nn


class NoAct(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(logits):
        return logits
