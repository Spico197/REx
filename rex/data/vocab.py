from typing import Callable, Iterable, Optional, List, Tuple

import numpy as np

from rex.utils.io import load_line_iterator, dump_iterable


def _convert_list_str_to_list(string: str, item_type: Callable):
    ret = []
    if string.startswith("["):
        string = string[1:]
    if string.endswith("]"):
        string = string[:-1]

    for item in string.split(","):
        ret.append(item_type(item.strip()))

    return ret


class Vocab(object):
    def __init__(
        self,
        pad: Optional[str] = "[PAD]",
        unk: Optional[str] = "[UNK]",
        include_pad: Optional[bool] = True,
        include_unk: Optional[bool] = True,
        embedding_dim: Optional[int] = 300,
        init_pad_unk_emb: Optional[bool] = False,
    ) -> None:
        self.pad = pad
        self.unk = unk
        self.pad_idx = None
        self.unk_idx = None

        self.token2id = {}
        self.id2token = {}
        self.weights = []

        if include_pad:
            token_idx = len(self.token2id)
            self.token2id[pad] = token_idx
            self.id2token[token_idx] = pad
            self.pad_idx = self.token2id[pad]
            if init_pad_unk_emb:
                self.weights.append(np.random.randn(embedding_dim).tolist())

        if include_unk:
            token_idx = len(self.token2id)
            self.token2id[unk] = token_idx
            self.id2token[token_idx] = unk
            self.unk_idx = self.token2id[unk]
            if init_pad_unk_emb:
                self.weights.append(np.random.randn(embedding_dim).tolist())

    def add(self, token: str, weights: Optional[List[float]] = None):
        if token not in self.token2id:
            token_id = len(self.token2id)
            self.token2id[token] = token_id
            self.id2token[token_id] = token

            if weights is not None:
                if token_id != len(self.weights):
                    raise ValueError(
                        "Vocab does not match weights, check `init_pad_unk_emb` option"
                    )
                self.weights.append(weights)

    def update(self, tokens: List[str]):
        for token in tokens:
            self.add(token)

    def convert_tokens_to_ids(self, tokens: List[str]):
        return list(map(lambda x: self.token2id.get(x, self.unk_idx), tokens))

    def convert_ids_to_tokens(self, ids: List[int]):
        if not all(idx in self.id2token for idx in ids):
            raise ValueError("Not all idx are in this vocab")
        return list(map(lambda x: self.id2token[x], ids))

    def update_convert_tokens_to_ids(self, tokens: List[str]):
        for token in tokens:
            self.add(token)
        return self.convert_tokens_to_ids(tokens)

    def encode(
        self, tokens: Iterable, max_seq_len: int, update: Optional[bool] = False
    ) -> Tuple[List[int]]:
        """convert tokens into ids by padding or cutting

        Args:
            update: whether to add tokens into vocab

        Returns:
            token_ids: token ids after encoding
            mask: padding mask
        """
        tokens = tokens[:max_seq_len]
        mask = [1] * len(tokens) + [0] * (max_seq_len - len(tokens))
        tokens = tokens + [self.pad] * (max_seq_len - len(tokens))
        if update:
            return self.update_convert_tokens_to_ids(tokens), mask
        else:
            return self.convert_tokens_to_ids(tokens), mask

    def clear_all(self):
        self.token2id.clear()
        self.id2token.clear()

    def __len__(self):
        return len(self.token2id)

    @property
    def size(self):
        return len(self)

    @classmethod
    def from_pretrained(
        cls, filepath, include_weights: Optional[bool] = False, **kwargs
    ):
        v = cls(**kwargs)
        for line in load_line_iterator(filepath):
            line = line.strip()
            if include_weights:
                token, weight_string = line.split("\t")
                weights = _convert_list_str_to_list(weight_string, float)
                v.add(token, weights)
        return v

    def save_pretrained(self, filepath, dump_weights: Optional[bool] = False):
        vocabs = []
        for token_id in range(self.size):
            if dump_weights:
                vocabs.append(f"{self.id2token[token_id]}\t{self.weights[token_id]}")
            else:
                vocabs.append(self.id2token[token_id])
        dump_iterable(vocabs, filepath)
