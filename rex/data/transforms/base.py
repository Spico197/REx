from typing import Dict, Iterable, Any

from rex.data.vocab import Vocab


class TransformBase(object):
    def __init__(self, max_seq_len) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.vocab = Vocab()

    def transform(self, lines: Iterable):
        raise NotImplementedError

    def predict_transform(self, strings: Iterable[str]):
        ret_data = []
        for string in strings:
            seq_len = min(len(string), self.max_seq_len)
            comp_len = max(0, (self.max_seq_len - seq_len))
            ret_data.append(
                {
                    "token_ids": self.vocab.convert_tokens_to_ids(list(string))[
                        :seq_len
                    ]
                    + comp_len * [self.vocab.pad_idx],
                    "mask": seq_len * [1] + comp_len * [0],
                }
            )
        return ret_data

    def __call__(self, *args: Any, **kwargs: Any):
        return self.transform(*args, **kwargs)
