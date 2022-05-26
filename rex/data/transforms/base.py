from typing import Any, Iterable


class TransformBase(object):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def transform(self, lines: Iterable):
        raise NotImplementedError

    def predict_transform(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any):
        return self.transform(*args, **kwargs)
