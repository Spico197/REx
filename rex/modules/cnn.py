import math
from typing import Iterable, Optional

import torch
import torch.nn as nn


class MultiKernelCNN(nn.Module):
    """
    Multi-kernel Convolutional Neural Network Module

    Args:
        in_channel: dimension of input features
        num_filters: number of filters in CNN
        kernel_sizes: kernel sizes, e.g. ``[1, 3, 5]``
        dropout: dropout rate

    Returns:
        torch.Tensor: Shape is :math:`x \\in \\mathbb{R}^{B \\times N \\cdot F}`,
            where :math:`B` is ``batch_size``,
            :math:`N` means the length of ``kernel_sizes``
            and :math:`F` is the number of CNN filters
    """

    def __init__(
        self,
        in_channel: int,
        num_filters: int,
        kernel_sizes: Iterable,
        dropout: Optional[float] = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(in_channel, num_filters, ks, padding=math.floor((ks - 1) / 2))
                for ks in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        batch_size = hidden.shape[0]
        # hidden: (batch_size, seq_len, in_channel) -> (batch_size, in_channel, seq_len)
        hidden = hidden.permute(0, 2, 1)
        # (len(kernel_sizes), batch_size, num_filters, seq_len)
        hidden = torch.stack([conv(hidden) for conv in self.convs], dim=0)
        # max pooling: (len(kernel_sizes), batch_size, num_filters)
        hidden = hidden.max(dim=-1)[0]
        # (batch_size, len(kernel_sizes), num_filters)
        hidden = hidden.permute(1, 0, 2)
        # (batch_size, len(kernel_sizes) * num_filters)
        hidden = hidden.reshape(batch_size, -1)
        hidden = self.dropout(hidden)
        return hidden
