from typing import Optional

import pytest
import torch


class RunIf(object):
    """Decorator for pytest mark.
    Mark as test only if conditions are satisfied.
    Ideas and code pieces borrow from PyTorch-Lightning_ (under Apache2.0 license).

    References:
        .. PyTorch-Lightning https://github.com/PyTorchLightning/pytorch-lightning/blob/d61371922b/tests/helpers/runif.py
    """

    def __new__(self, *args, min_gpus: Optional[int] = 0, **kwargs):
        conditions = []
        reasons = []

        if min_gpus > 0:
            conditions.append(torch.cuda.device_count() < min_gpus)
            reasons.append(f"GPUs >= {min_gpus}")

        reasons = [rs for cond, rs in zip(conditions, reasons) if cond]
        return pytest.mark.skipif(
            *args,
            condition=any(conditions),
            reason=f"Requires: [{' + '.join(reasons)}]",
            **kwargs,
        )
