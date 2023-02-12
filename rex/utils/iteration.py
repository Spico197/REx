from typing import Any, Iterable, Iterator, List, Union


def flatten_all_iter(iterable: Iterable) -> Iterator:
    for elem in iterable:
        if not isinstance(elem, list):
            yield elem
        else:
            yield from flatten_all_iter(elem)


def windowed_queue_iter(
    queue: List[Any],
    window: int,
    stride: Union[int, None] = None,
    drop_last: bool = False,
) -> Iterator:
    if not stride:
        stride = window
    if len(queue) <= window:
        yield queue
    else:
        if drop_last:
            max_limit = len(queue) - window + 1
        else:
            max_limit = len(queue)
        for i in range(0, max_limit, stride):
            yield queue[i : i + window]
