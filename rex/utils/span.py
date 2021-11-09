import copy
from typing import Iterable, Optional

import torch
import numpy as np


def find_closest_span_pairs(
    head: Iterable, tail: Iterable, backtrace: Optional[bool] = True
):
    """
    Find all span pairs.

    Args:
        head: list of start position predictions, either 1 or 0
        tail: list of end position predictions, either 1 or 0
        backtrace: if there are more tail predictions than head predictions,
            then backtrace to find a closest head position to get a span pair

    Examples:
        >>> head = torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.long)
        >>> tail = torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.long)
        >>> find_closest_span_pairs(head, tail, backtrace=False)
        [(0, 1), (3, 3), (6, 6)]
        >>> find_closest_span_pairs(head, tail, backtrace=True)
        [(0, 1), (3, 3), (6, 6), (3, 5)]
    """
    if isinstance(head, torch.Tensor):
        head = head.detach().cpu()
    if isinstance(tail, torch.Tensor):
        tail = tail.detach().cpu()
    head_valid_poses = np.where(head == 1)[0]
    tail_valid_poses = np.where(tail == 1)[0]
    tail_used_poses = {pos: False for pos in tail_valid_poses.tolist()}

    pairs = []
    for head_i in head_valid_poses:
        tail_js = tail_valid_poses[tail_valid_poses >= head_i]
        if len(tail_js) > 0:
            tail_j = tail_js[0]
            tail_used_poses[tail_j] = True
            pairs.append((head_i, tail_j))

    # some remained tails
    if backtrace:
        for tail_j in tail_used_poses:
            if tail_used_poses[tail_j] is False:
                head_is = head_valid_poses[head_valid_poses <= tail_j]
                if len(head_is) > 0:
                    head_i = head_is[-1]
                    pairs.append((head_i, tail_j))
    return pairs


def find_closest_span_pairs_with_index(
    heads: Iterable, tails: Iterable, backtrace: Optional[bool] = True
):
    """
    Find all possible pairs with indexes,
    useful for object discoveries with class idx.

    Args:
        heads: batch of torch.Tensor
        tails: batch of torch.Tensor
        backtrace: if there are more tail predictions than head predictions,
            then backtrace to find a closest head position to get a span pair

    Examples:
        >>> heads = torch.tensor([[1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1]], dtype=torch.long)
        >>> tails = torch.tensor([[0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0]], dtype=torch.long)
        >>> find_closest_span_pairs(heads, tails, backtrace=False)
        [(0, 0, 1), (0, 3, 3), (0, 6, 6), (1, 0, 1), (1, 3, 5)]
        >>> find_closest_span_pairs(heads, tails, backtrace=True)
        [(0, 0, 1), (0, 3, 3), (0, 6, 6), (0, 3, 5), (1, 0, 1), (1, 3, 5)]
    """
    results = []
    for idx, (head, tail) in enumerate(zip(heads, tails)):
        pairs = find_closest_span_pairs(head, tail, backtrace=backtrace)
        for pair in pairs:
            results.append((idx, pair[0], pair[1]))
    return results
