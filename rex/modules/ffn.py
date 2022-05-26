from typing import Iterable, Optional

import torch
import torch.nn as nn

from rex.modules.dropout import SharedDropout


class FFN(nn.Module):
    """
    Multi-layer feed-forward neural networks

    Args:
        input_dim: input dimension
        output_dim: output dimension
        mid_dims: middle dimensions, if None, FFN is equals to `nn.Linear` with dropout
        dropout: dropout rate
        act_fn: activation function (module class without instantiated)

    Input:
        hidden: hidden states
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mid_dims: Optional[Iterable[int]] = None,
        dropout: Optional[float] = 0.5,
        act_fn: Optional[nn.Module] = nn.ReLU(),
    ):
        super().__init__()

        if mid_dims is None:
            self.ffn = nn.Sequential(
                nn.Linear(input_dim, output_dim), nn.Dropout(dropout)
            )
        else:
            mid_dims = list(mid_dims)
            mid_dims.insert(0, input_dim)
            mid_dims.append(output_dim)
            len_mid_dims = len(mid_dims)
            modules = []
            for i in range(len_mid_dims - 2):
                modules.extend(
                    [
                        nn.Linear(mid_dims[i], mid_dims[i + 1]),
                        nn.Dropout(dropout),
                        act_fn,
                    ]
                )
            modules.append(nn.Linear(mid_dims[-2], mid_dims[-1]))
            self.ffn = nn.Sequential(*modules)

    def forward(self, hidden):
        return self.ffn(hidden)


class MLP(nn.Module):
    """Implements Multi-layer Perception."""

    def __init__(
        self,
        input_size,
        output_size,
        mid_size=None,
        num_mid_layer=1,
        act_fn=torch.relu,
        dropout=0.1,
    ):
        super(MLP, self).__init__()

        assert num_mid_layer >= 1
        if mid_size is None:
            mid_size = input_size

        self.act_fn = act_fn
        self.input_fc = nn.Linear(input_size, mid_size)
        self.out_fc = nn.Linear(mid_size, output_size)
        if num_mid_layer > 1:
            self.mid_fcs = nn.ModuleList(
                nn.Linear(mid_size, mid_size) for _ in range(num_mid_layer - 1)
            )
        else:
            self.mid_fcs = []
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act_fn(self.input_fc(x)))
        for mid_fc in self.mid_fcs:
            x = self.dropout(self.act_fn(mid_fc(x)))
        x = self.out_fc(x)
        return x


class SharedDropoutMLP(nn.Module):
    r"""
    Applies a linear transformation together with a non-linear activation to the incoming tensor:
    :math:`y = \mathrm{Activation}(x A^T + b)`
    Args:
        n_in (~torch.Tensor):
            The size of each input feature.
        n_out (~torch.Tensor):
            The size of each output feature.
        dropout (float):
            If non-zero, introduce a :class:`SharedDropout` layer on the output with this dropout ratio. Default: 0.
        activation (bool):
            Whether to use activations. Default: True.
    """

    def __init__(self, n_in, n_out, dropout=0, activation=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = (
            nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        )
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                The size of each input feature is `n_in`.
        Returns:
            A tensor with the size of each output feature `n_out`.
        """

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x
