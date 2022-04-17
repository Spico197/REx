from typing import Iterable, Iterator


def flatten_all(iterable: Iterable) -> Iterator:
    for elem in iterable:
        if not isinstance(elem, list):
            yield elem
        else:
            yield from flatten_all(elem)
