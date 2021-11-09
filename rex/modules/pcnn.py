import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PiecewiseCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_filters,
        kernel_size,
        dropout: Optional[float] = 0.5,
        activation_function: Optional[Callable] = torch.tanh,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            num_filters,
            kernel_size,
            padding=math.floor((kernel_size - 1) / 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.act = activation_function

        # mask operation for pcnn
        masks = torch.tensor(
            [[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]], dtype=torch.float
        )
        self.mask_embedding = nn.Embedding.from_pretrained(masks, freeze=True)

    def forward(self, input_rep, mask):
        # mask: (batch_size, seq_len)
        # input_rep: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size, seq_len)
        x = input_rep.permute(0, 2, 1)
        # x: (batch_size, num_filters, seq_len)
        x = self.conv(x)

        # mask_embed: (batch_size, seq_len, 3)
        mask_embed = self.mask_embedding(mask)
        # mask_embed: (batch_size, 1, seq_len, 3)
        mask_embed = mask_embed.unsqueeze(dim=1)
        # x: (batch_size, num_filters, seq_len, 3)
        x = x.unsqueeze(-1) + mask_embed
        # x: (batch_size, num_filters, 3)
        x = torch.max(x, dim=2)[0] - 100
        # x: (batch_size, 3*num_filters)
        x = x.view(x.shape[0], -1)

        x = torch.tanh(x)
        x = self.dropout(x)
        return x
