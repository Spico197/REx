from typing import Callable, Iterable, Optional, List, Tuple

import numpy as np

from rex.utils.io import dump_iterable, load_embedding_file


def _convert_list_str_to_list(string: str, item_type: Callable):
    ret = []
    if string.startswith("["):
        string = string[1:]
    if string.endswith("]"):
        string = string[:-1]

    for item in string.split(","):
        ret.append(item_type(item.strip()))

    return ret


def get_pad_mask(token_len, max_len, token_mask=1, pad_mask=0):
    mask = [token_mask] * token_len + [pad_mask] * (max_len - token_len)
    return mask


def get_pad_token(tokens, max_len, pad_token):
    tokens_len = len(tokens)
    tokens = tokens[:max_len]
    tokens = tokens + [pad_token] * (max_len - tokens_len)
    return tokens


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

            if token == self.pad:
                self.pad_idx = token_id
            elif token == self.unk:
                self.unk_idx = token_id

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
        tokens = get_pad_token(tokens, max_seq_len, self.pad)
        mask = get_pad_mask(len(tokens), max_seq_len, 1, 0)
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
        tokens, token2vec = load_embedding_file(filepath)
        for token in tokens:
            vec = None
            if include_weights:
                vec = token2vec[token]
            v.add(token, weights=vec)
        return v

    def save_pretrained(self, filepath, dump_weights: Optional[bool] = False):
        vocabs = []
        for token_id in range(self.size):
            if dump_weights:
                weight_str = " ".join(map(str, self.weights[token_id]))
                vocabs.append(f"{self.id2token[token_id]} {weight_str}")
            else:
                vocabs.append(self.id2token[token_id])
        dump_iterable(vocabs, filepath)
