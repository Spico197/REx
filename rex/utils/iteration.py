from typing import Any, Iterable, Iterator, List


def flatten_all_iter(iterable: Iterable) -> Iterator:
    for elem in iterable:
        if not isinstance(elem, list):
            yield elem
        else:
            yield from flatten_all_iter(elem)


def windowed_queue_iter(queue: List[Any], window: int) -> Iterator:
    if len(queue) <= window:
        yield queue
    else:
        for i in range(0, len(queue), window):
            yield queue[i : i + window]
